from models.model_builder import ModelBuilder
from utils.load_semantic_datasets import SemanticGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
from utils.metrics import CityEvalMIoU, MIoU
from utils.get_flops import get_flops
from utils.predict_utils import color_map

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",      type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=3)
parser.add_argument("--image_size",      type=tuple,
                    help="Model image size (input resolution H,W)", default=(224, 224))
parser.add_argument("--dataset_dir",     type=str,
                    help="Dataset directory", default='./datasets/')
parser.add_argument("--dataset_name",     type=str,
                    help="Dataset directory", default='full_semantic')
parser.add_argument("--checkpoint_dir",  type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_path",     type=str,
                    help="Saved model weights directory", default='0802/_0802_Test_os_32_best_iou.h5')

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
                                       batch_size=args.batch_size, dataset_name=args.dataset_name)
    dataset = dataset_config.get_testData(valid_data=dataset_config.valid_data)
    test_steps = dataset_config.number_valid // args.batch_size

    # Model build and load pre-trained weights
    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_path)
    model.summary()

    # Model warm up
    _ = model.predict(tf.zeros((1, args.image_size[0], args.image_size[1], 3)))

    # Set evaluate metrics and Color maps
    if args.dataset_name == 'cityscapes':
        miou = CityEvalMIoU(args.num_classes+1)
        color_map = dataset_config.cityscapes_tools.trainabel_color_map
    else:
        miou = MIoU(args.num_classes)
    
    # Set plot configs
    rows = 1
    cols = 2
    batch_idx = 0
    avg_duration = 0
    batch_index = 0

    # Predict
    for x, gt, original_img in tqdm(dataset, total=test_steps):
        # Check inference time
        start = time.process_time()
        pred = model.predict_on_batch(x)
        duration = (time.process_time() - start)

        # Argmax prediction
        pred = tf.math.argmax(pred, axis=-1, output_type=tf.int32)

        

        for i in range(args.batch_size):
            # Calculate metrics
            miou.update_state(gt[i], pred[i])
            metric_result = miou.result().numpy()


        if args.visualize:
            for i in range(args.batch_size):
                r = pred[i]
                g = pred[i]
                b = pred[i]

                for j in range(args.num_classes):
                    r = tf.where(tf.equal(r, j), color_map[j][0], r)
                    g = tf.where(tf.equal(g, j), color_map[j][1], g)
                    b = tf.where(tf.equal(b, j), color_map[j][2], b)

                r = tf.expand_dims(r, axis=-1)
                g = tf.expand_dims(g, axis=-1)
                b = tf.expand_dims(b, axis=-1)

                rgb_img = tf.concat([r, g, b], axis=-1)

                tf.keras.preprocessing.image.save_img(args.result_dir + str(batch_index)+'.png', rgb_img)
                batch_index += 1

        avg_duration += duration
        batch_idx += 1

    print('Model FLOPs {0}'.format(get_flops(model=model, batch_size=1)))
    print('avg inference time : {0}sec.'.format((avg_duration / dataset_config.number_valid)))
    print('Image size : {0},  MIoU : {1}'.format(args.image_size, metric_result))