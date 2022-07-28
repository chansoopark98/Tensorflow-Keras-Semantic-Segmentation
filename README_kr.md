<h1>End-to-End Semantic Segmentation</h1>

> All about Tensorflow/Keras semantic segmentation


## Tensorflow/Keras를 활용한 semantic segmentation repository
<br>

<center>
<img src="https://img.shields.io/github/issues/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
<img src="https://img.shields.io/github/forks/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
<img src="https://img.shields.io/github/stars/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
<img src="https://img.shields.io/github/license/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FTensorflow-Keras-Realtime-Segmentation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23C41010&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)



<br>

<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
<img src ="https://img.shields.io/badge/Tensorflow-FF6F00.svg?&style=for-the-badge&logo=Tensorflow&logoColor=white"/>
<img src ="https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>
<img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
<img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>

<br>

</center>

![main_image_1](https://user-images.githubusercontent.com/60956651/181407216-63498ca5-7668-4188-853b-c48506534b9e.png)

<center>Cityscapes 이미지 분할 결과 (with ignore index)</center>

<br>

![166](https://user-images.githubusercontent.com/60956651/181407706-1d2ba5cd-fe9f-419f-aa03-e44e6e77a40e.png)

<center>Cityscapes 이미지 분할 결과 (without ignore index)</center>

<br>


### 지원하는 기능
- 데이터 전처리
- Train
- Evaluate
- Predict real-time
- TensorRT 변환
- Tensorflow docker serving

<br>

### **Use library:** 
- Tensorflow
- Tensorflow-datasets
- Tensorflow-addons
- Tensorflow-serving
- Keras
- OpenCV python
- gRPC

### **Options:** Distribute training, Custom Data
### **Models:** DDRNet-23-Slim, Eff-DeepLabV3+, Eff-DeepLabV3+(light-weight), MobileNetV3-DeepLabV3+


<br>
<hr/>

# Table of Contents

 ## 1. [Models](#1-models-1)
 ## 2. [Dependencies](#2-dependencies-1)
 ## 3. [Preparing datasets](#3-preparing-datasets-1)
 ## 4. [Train](#4-train-1)
 ## 5. [Eval](#5-eval-1)
 ## 6. [Predict](#6-predict-1)
 ## 7. [Convert TF-TRT](#7-convert-tf-trt-1)
 ## 8. [Tensorflow serving](#8-tensorflow-serving-1)

<br>
<hr/>

# 1. Models

<table border="0">
<tr>
    <tr>
        <td>
        Model name
        </td>
        <td>
        Params(Million)
        </td>
        <td>
        Resolution(HxW)
        </td>
        <td>
        Inference time(ms)
        </td>
    </tr>
    <tr>
        <td>
        DDRNet-23-slim
        </td>
        <td>
        0.5m
        </td>
        <td>
        640x480
        </td>
        <td>
        20ms
        </td>
    </tr>
</tr>
</table>

## Loss
1. **CrossEntropy** <br>

    ![image](https://user-images.githubusercontent.com/60956651/163329997-a0b8d85d-c98a-401b-abba-0d65b4a8e303.png)

<br>

2. **FocalLoss** <br>

    
    <img src="https://user-images.githubusercontent.com/60956651/163326665-a4a55c98-b2b7-4822-a6e5-2379e6023b8a.png"  width="500" height="370">

    <br>

    ### $FL (pt) = -αt(1-  pt)γ log  log(pt)$

    <br>

    When an example is misclassified and pt is small, the modulating factor is near 1 and the loss is unaffected. <br><br>
    As pt→  1, the factor goes to 0 and the loss for well-classified examples is down weighed. <br><br>
    The focusing parameter
    γ smoothly adjusts the rate at which easy examples are down-weighted. <br><br>
    As is increased, the effect of modulating factor is likewise increased. (After a lot of experiments and trials, researchers have found γ = 2 to work best)

    <br>

3. **BinaryCrossEntropy**
    <br>

    ### $BCE(x)=−1N∑i=1Nyilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))$

    <br>

4. **BinaryFocalLoss**
    <br>

    ### $L(y,p^)=−αy(1−p^ )γlog(p^)−(1−y)p^γlog(1−p^)$
    
    where

    y∈{0,1} is a binary class label, <br>

    p^∈[0,1] is an estimate of the probability of the positive class, <br>

    γ is the focusing parameter that specifies how much higher-confidence correct predictions contribute to the overall loss (the higher the γ, the higher the rate at which easy-to-classify examples are down-weighted). <br><br>

    α is a hyperparameter that governs the trade-off between precision and recall by weighting errors for the positive class up or down (α=1 is the default, which is the same as no weighting),

    <br>
    
<br>
<hr/>

# 2. Dependencies

본 레포지토리의 종속성은 다음과 같습니다.

<table border="0">
<tr>
    <tr>
        <td>
        OS
        </td>
        <td>
        Ubuntu 18.04
        </td>
    </tr>
    <tr>
        <td>
        TF version
        </td>
        <td>
        2.9.1
        </td>
    </tr>
    <tr>
        <td>
        Python version
        </td>
        <td>
        3.8.13
        </td>
    </tr>
    <tr>
        <td>
        CUDA
        </td>
        <td>
        11.1
        </td>
    </tr>
    <tr>
        <td>
        CUDNN
        </td>
        <td>
        cuDNN v8.1.0 , for CUDA 11.1
        </td>
    </tr>
        <tr>
        <td>
        TensorRT version
        </td>
        <td>
        7.2.2.3
        </td>
    </tr>
    </tr>
        <tr>
        <td>
        Docker
        </td>
        <td>
        Docker 20.10.17
        </td>
    </tr>
</table>
<br>
<hr/>

학습 및 평가를 위해 **Anaconda(miniconda)** 가상환경에서 패키지를 다운로드 합니다.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<br>
<hr/>

# 3. Preparing datasets

프로그램에 필요한 **Dataset**은 **Tensorflow Datasets** 라이브러리([TFDS](https://www.tensorflow.org/datasets/catalog/overview))를 사용합니다. 

<br>

## **Custom dataset labeling process**
Custom data image labeling 작업은 **CVAT**(https://github.com/openvinotoolkit/cvat)이라는 툴을 사용하여 작업하였습니다.

Labeling 작업이 완료되고 **CVAT**에서 dataset의 export format을 **Segmentation mask 1.1 format**으로 생성합니다.

생성된 데이터셋의 **labelmap.txt**에서 각 클래스별 RGB값을 확인할 수 있습니다.
* 어떻게 사용하나요?
    1. CVAT tool을 사용하여 semantic data(mask) 레이블링
    2. 데이터 증강
        * 이미지 시프트
        * 이미지 블러링
        * 이미지 회전
        * 마스크 영역 이미지 변환 .. etc

<br>
첫번째, foreground가 없는 이미지의 경우 CVAT에서 자동으로 레이블을 생성하지 않습니다.
아래와 같이 foreground object가 없을 때를 가정하여 zero label을 생성합니다.
<br>

<br>

    cd data_augmentation
    python make_blank_label.py

<br>

두번째, 

    python augment_data.py
    
    아래의 옵션들의 경로를 지정하여 증강을 수행합니다.

    --rgb_path RGB_PATH   raw image path
    --mask_path MASK_PATH
                            raw mask path
    --obj_mask_path OBJ_MASK_PATH
                            raw obj mask path
    --label_map_path LABEL_MAP_PATH
                            CVAT's labelmap.txt path
    --bg_path BG_PATH     bg image path, Convert raw rgb image using mask area
    --output_path OUTPUT_PATH
                            Path to save the conversion result

<br>

### **Caution!**
코드 하단의 __main__:에서 직접 증강할 옵션들을 선택할 수 있습니다. 이 부분을 수정하여 원하는 증강 방법에 맞게 변경해보세요.

<br>

## **Convert TFDS dataset**

생성된 Semantic label을 tf.data format으로 변환하기 위해 tensorflow datasets 라이브러리를 사용합니다.<br>

증강이 적용된 RGB 이미지와 semantic label이 저장된 이미지를 다음과 같은 폴더로 이동시킵니다.

<br>


    └── dataset 
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

해당 디렉토리를 'full_semantic.zip'으로 압축시킵니다.

    zip full_semantic.zip ./*

압축이 완료되면 해당 경로와 같이 설정되어야 합니다.

    
    └──full_semantic.zip
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

그리고 나서, full_semantic.zip을 아래와 같은 폴더 구조를 생성한 후 이동시킵니다.

    /home/$USER/tensorflow_datasets/downloads/manual

마지막으로 데이터셋을 빌드합니다.
    
    cd hole-detection/full_semantic/
    tfds build

    # if build is successfully
    cd -r home/$USER/tensorflow_datasets/
    cp full_semantic home/$USER/hole-detection/datasets/
<br>

### **Caution!**
**augment_data.py**의 작업 결과물은 기본적으로 RGB, MASK, VIS_MASK 세 개의 경로로 구성됩니다.<br>
**VIS_MASK**는 실제 사용될 레이블은 아니며, 시각적으로 확인하기위한 용도이니 아래 작업에서 사용하지 마세요. <br>
<br>

<hr/>

# 4. Train

학습하기전 tf.data의 메모리 할당 문제로 인해 TCMalloc을 사용하여 메모리 누수를 방지합니다.

    1. sudo apt-get install libtcmalloc-minimal4
    2. dpkg -L libtcmalloc-minimal4

    2번을 통해 설치된 TCMalloc의 경로를 저장합니다


## Training semantic segmentation

**How to RUN?**
    
Single gpu

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

Mutli gpu

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py --multi_gpu


### **Caution!**
본 레포지토리는 single-GPU, multi-GPU 환경에서의 학습 및 추론을 지원합니다. <br>
Single-GPU 사용 시, GPU 번호를 설정하여 사용할 수 있습니다. <br>
python train.py --help를 살펴보시고 학습에 필요한 설정값을 argument 인자값으로 추가해주세요. <br>

<br>
<hr>

# 5. Eval
Training 이후 모델의 정확도 평가 및 추론 속도를 계산합니다. <br>
<br>
계산 항목 : FLOPs, MIoU metric, Average inference time
<br>

    python eval.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

<br>
추론 결과를 확인하려는 경우 --visualize 인자를 추가해주세요.



<hr>

# 6. Predict
Web-camera 또는 저장된 비디오를 실시간으로 추론할 수 있습니다. <br>
<br>

**비디오 추론의 경우**

    python predict_video.py

<br>

**Web-cam 실시간 추론**

    python predict_realtime.py


<br>

추론 결과를 확인하려는 경우 **--visualize** 인자를 추가해주세요.

<br>
<hr>

# 7. Convert TF-TRT
고속 추론이 가능하도록 TF-TRT 변환 기능을 제공합니다.
변환에 앞서 tensorRT를 설치합니다.


## 7.1 Install CUDA, CuDNN, TensorRT files

<br>

현재 작성된 코드 기준으로 사용된 CUDA 및 CuDNN 그리고 TensorRT version은 다음과 같습니다. <br>
클릭 시 설치 링크로 이동합니다. <br>
CUDA 및 CuDNN이 사전에 설치가 완료된 경우 생략합니다.

<br>

### CUDA : **[CUDA 11.1](https://www.tensorflow.org/datasets/catalog/overview)**
### CuDNN : **[CuDNN 8.1.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz)**
### TensorRT : **[TensorRT 7.2.2.3](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.2/tars/tensorrt-7.2.2.3.ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz)**

<br>

## 7.2 Install TensorRT
<br>

가상 환경을 활성화합니다. (Anaconda와 같이 가상환경을 사용하지 않은 경우 생략합니다)
    
    conda activate ${env_name}

<br>

TensorRT를 설치한 디렉토리로 이동하여 압축을 해제하고 pip를 업그레이드 합니다.

    tar -xvzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
    pip3 install --upgrade pip

편집기를 이용하여 배시 쉘에 접근하여 환경 변수를 추가합니다.

    sudo gedit ~/.bashrc
    export PATH="/usr/local/cuda-11.1/bin:$PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/park/TensorRT-7.2.2.3/onnx_graphsurgeon
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/park/TensorRT-7.2.2.3/lib"

TensorRT 파이썬 패키지를 설치합니다.

    cd python
    python3 -m pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl

    cd ../uff/
    python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

    cd ../graphsurgeon
    python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

    cd ../onnx_graphsurgeon
    python3 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl

terminal을 열어서 설치가 잘 되었는지 확인합니다.

![test_python](https://user-images.githubusercontent.com/60956651/181165197-6a95119e-ea12-492b-9587-a0c5badc73be.png)

<br>

## 7.3 Convert to TF-TensorRT

TF-TRT 변환 작업 전 사전 학습된 graph model **(.pb)**이 필요합니다. <br>
Graph model이 없는 경우 **7.3.1** 절차를 따르고, 있는 경우에는 **7.3.2**로 넘어가세요.


- ### 7.3.1 Graph model이 없는 경우

    본 레포지토리에서 **train.py**를 통해 학습된 가중치가 있는 경우 graph model로 변환하는 기능을 제공합니다.

    **train.py**에서 **--saved_model** argument로 그래프 저장 모드를 활성화합니다. 그리고 학습된 모델의 가중치가 저장된 경로를 추가해줍니다.

        python train.py --saved_model --saved_model_path='your_model_weights.h5'

    변환된 graph model의 기본 저장 경로는 **'./checkpoints/export_path/1'** 입니다.

    ![saved_model_path](https://user-images.githubusercontent.com/60956651/181168185-376880d3-b9b8-4ea7-8c76-be8b498e34b1.png)

    <br>

- ### 7.3.2 Converting

    **(.pb)** 파일이 존재하는 경우 아래의 스크립트를 실행하여 변환 작업을 수행합니다.

        python convert_to_tensorRT.py ...(argparse options)

    TensorRT 엔진을 통해 모델을 변환합니다. 고정된 입력 크기를 바탕으로 엔진을 빌드하니 스크립트 실행 전 **--help** 인자를 확인해주세요.

    <br>
    
    아래와 같은 옵션을 제공합니다. <br>

    **모델 입력 해상도** (--image_size), **.pb 파일 디렉토리 경로** (input_saved_model_dir) <br>
    
    **TensorRT 변환 모델 저장 경로** (output_saved_model_dir), **변환 부동소수점 모드 설정** (floating_mode)

    <br>

<hr>

# 8. Tensorflow serving

사전 학습된 graph model (.pb) 또는 TensorRT 엔진으로 빌드된 모델을 서비스하는 기능을 제공합니다. <br>

Tensorflow serving은 도커 가상 환경내에서 추론 서비스를 제공하는 도구입니다.

작업에 앞서 현재 운영체제 version에 맞는 도커를 설치합니다. (https://docs.docker.com/engine/install/ubuntu/) <br>


    # Ubuntu 18.04 docker install

    # 1. Preset
    sudo apt update
    sudo apt install apt-transport-https ca-certificates curl software-properties-common

    # 2. Add docker repository keys
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"

    # 3. Install
    sudo apt update
    sudo apt install docker-ce

도커 설치와 모델 파일이 준비되어있다면 바로 실행 가능합니다. 실행에 앞서 Serving server를 설정 옵션을 살펴봅니다. 

    docker run 
	--runtime=nvidia # 도커에서 nvidia-gpu를 사용하기 위한 설정
	-t # tty 사용
	-p 8500:8500 # 도커 환경에서 open할 포트주소
	--rm # 도커 컨테이너가 사용되지 않을 경우 자동으로 삭제
	-v "model_path:/models/test_model2" # {1}:{2} -> {1}은 .pb파일이 있는 경로입니다.
                             {2}는 도커에서 해당 모델이 배포될 경로 (test_model2 이름을 활용하여 request 요청)
	-e MODEL_NAME=test_model2 # gRPC, REST API에 호출될 모델 이름
	-e NVIDIA_VISIBLE_DEVICES="0" # 사용할 gpu 번호
	-e LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 # TensorRT 엔진 빌드를 위해 cuda-11.1 환경 변수 지정 (TensorRT 엔진 7.2.2.3)
	-e TF_TENSORRT_VERSION=7.2.2 tensorflow/serving:2.6.2-gpu # TensorRT 버전 설정 및 해당 버전에 맞는 tensorflow-gpu 버전 설치
	--port=8500 # Serving 시 사용할 포트번호 (도커 포트 설정과 동일하게 해줘야 함)


명령어 끝에 **--help** 인자로 추가 정보를 확인할 수 있습니다. <br>

Tensorflow-serving 서버에 접근하여 추론 요청을 하는 예제는 **tf_serving_sample.py** 를 참고해주세요. <br>


# Reference
<hr>

1. [DDRNet : https://github.com/ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet)
2. [CSNEt-seg https://github.com/chansoopark98/CSNet-seg](https://github.com/chansoopark98/CSNet-seg)