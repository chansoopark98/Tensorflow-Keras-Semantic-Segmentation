from models.model_builder import ModelBuilder
from utils.load_semantic_datasets import SemanticGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrics import MIoU

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",      type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=2)
parser.add_argument("--image_size",      type=tuple,
                    help="Model image size (input resolution H,W)", default=(320, 240))
parser.add_argument("--dataset_dir",     type=str,
                    help="Dataset directory", default='./datasets/')
parser.add_argument("--checkpoint_dir",  type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_path",     type=str,
                    help="Saved model weights directory", default='your_model_weights.h5')

# Prediction results visualize options
parser.add_argument("--visualize",  help="Whether to image and save inference results", action='store_true')
parser.add_argument("--result_dir",      type=str,
                    help="Test result save directory", default='./results/')

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
    model.load_weights(args.checkpoint_dir + args.weight_path)
    model.summary()

    # Model warm up
    model.predict(tf.zeros((1, args.image_size[0], args.image_size[1], 3)))

    # Set evaluate metrics
    miou = MIoU(args.num_classes)
    
    # Set plot config
    rows = 2
    cols = 2
    batch_idx = 0
    avg_duration = 0

    # Predict
    for x, gt, original_img in tqdm(dataset, total=test_steps):
        # Check inference time
        start = time.process_time()
        pred = model.predict(x)
        duration = (time.process_time() - start)

        # Calculate metrics
        miou.update_state(gt, pred)
        metric_result = miou.result().numpy()


        if args.visualize:
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
                plt.savefig(args.result_dir + save_name, dpi=300)
                plt.close()

        avg_duration += duration
        batch_idx += 1

    print(f"avg inference time : {(avg_duration / dataset_config.number_valid)}sec.")
    print('Image size : {0},  MIoU : {1}'.format(args.image_size, metric_result))