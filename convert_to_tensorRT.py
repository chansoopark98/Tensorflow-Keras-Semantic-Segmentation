from model_configuration import ModelConfiguration
import argparse


parser = argparse.ArgumentParser()

# Set Convert to TensorRT
parser.add_argument("--saved_model_path", type=str,   help="저장된 모델 가중치 경로",
                    default='Your_model_weights.h5')

args = parser.parse_args()

if __name__ == '__main__':
    model = ModelConfiguration(args=args)
    model.convert_to_trt()
    