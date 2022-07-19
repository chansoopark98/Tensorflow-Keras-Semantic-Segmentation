from models.model_builder import ModelBuilder
from utils.load_semantic_datasets import SemanticGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=2)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(320, 240))
parser.add_argument("--dataset_dir",    type=str,
                    help="Dataset directory", default='./datasets/')
parser.add_argument("--result_dir", type=str,
                    help="Test result save directory", default='./results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/0719/_0719_B8_E200_LR0.001_320-240_MultiGPU_sigmoid_activation_EFFV2S_best_iou.h5')

args = parser.parse_args()

if __name__ == '__main__':
    # Create result plot image path
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Configuration test(valid) datasets
    dataset_config = SemanticGenerator(data_dir=args.dataset_dir, image_size=args.image_size,
    batch_size=args.batch_size, dataset_name='full_semantic')
    dataset = dataset_config.get_testData(valid_data=dataset_config.valid_data)

    test_steps = dataset_config.number_valid // args.batch_size

    # Model build and load pre-trained weights
    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    # Model warm up
    model.predict(tf.zeros((1, args.image_size[0], args.image_size[1], 3)))
    
    # Set plot config
    rows = 2
    cols = 2
    batch_idx = 0
    avg_duration = 0

    # Predict
    for x, gt, original_img in tqdm(dataset, total=test_steps):
        start = time.process_time()

        pred = model.predict(x)
        
        duration = (time.process_time() - start)

        for i in range(args.batch_size):
            semantic_pred = pred[i, :, :, :args.num_classes]
            confidence_pred = pred[i, :, :, args.num_classes:]
            
            original_mask = gt[i, :, :, 0]

            semantic_pred = tf.argmax(semantic_pred, axis=-1)

            original_img = original_img[i]

            fig = plt.figure()
            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(original_img)
            ax0.set_title('original_img')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 2)
            ax0.imshow(original_mask)
            ax0.set_title('original_mask')
            ax0.axis("off")

            
            ax0 = fig.add_subplot(rows, cols, 3)
            ax0.imshow(semantic_pred)
            ax0.set_title('semantic')
            ax0.axis("off")


            ax0 = fig.add_subplot(rows, cols, 4)
            ax0.imshow(confidence_pred)
            ax0.set_title('confidence')
            ax0.axis("off")
            # plt.show()

            save_name = 'idx_{0}_batch_{1}'.format(batch_idx, i) + '.png'
            plt.savefig(args.result_dir + save_name, dpi=400)
            plt.close()

        avg_duration += duration
        batch_idx += 1

    print(f"avg inference time : {(avg_duration / dataset_config.number_valid)}sec.")