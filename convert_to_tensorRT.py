from utils.convert_trt import convert_to_trt
import argparse


parser = argparse.ArgumentParser()

# Set Convert to TensorRT
parser = argparse.ArgumentParser()
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(640, 360))
parser.add_argument("--input_saved_model_dir",    type=str,
                    help="Dataset directory", default='./checkpoints/export_path/1/')
parser.add_argument("--output_saved_model_dir", type=str,
                    help="Test result save directory", default='./checkpoints/export_path_trt/1/')
parser.add_argument("--floating_mode", type=str,
                    help="Floating mode to be converted (FP32 or FP16)", default='FP16')
args = parser.parse_args()



if __name__ == '__main__':
    convert_to_trt(image_size=args.image_size,
     saved_model_path=args.input_saved_model_dir, output_model_path=args.output_saved_model_dir)
    