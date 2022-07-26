<h1>Realtime segmentation with custom datasets</h1>

> Binary/Semantic segmentation with Custom data

이 저장소는 **Custom dataset을 사용한 segmentation**을 위해 데이터 레이블링부터 실시간 추론까지 구현하였습니다.
 &nbsp; **(Binary segmentation, Semantic segmentation)**

<br>

### **Keyword:** Salient Object Detection, Binary Segmentation, Semantic Segmentation, Deep learning, Lightweight, distribute training

<br>

### **Use library:** Tensorflow, Keras, OpenCV
### **Options:** Distribute training, Custom Data
### **Models:** DDRNet-23-Slim, Eff-DeepLabV3+, Eff-DeepLabV3+(light-weight), MobileNetV3-DeepLabV3+


<br>
<hr/>

## Table of Contents

 1. [Models](#Models)
 2. [Preferences](#Preferences)
 3. [Preparing datasets](#Preparing-datasets)
 4. [Train](#Train)
 5. [Eval](#Eval)
 6. [Predict](#Predict)

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

# 2. Preferences

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
        2.6.2
        </td>
    </tr>
    <tr>
        <td>
        Python version
        </td>
        <td>
        3.8.12
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
        GPU
        </td>
        <td>
        NVIDIA RTX3090 24GB * 2
        </td>
    </tr>
</table>

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
* Semantic label
    1. 데이터 생성 및 시멘틱 레이블링
    2. 데이터 증강
        * 이미지 시프트
        * 이미지 블러링
        * 이미지 회전
        * 마스크 영역 이미지 변환


## **Generate Binary mask label**


utils/generate_binary_mask.py를 실행하여 생성합니다.

    cd utils
    python generate_binary_mask.py --image_path='./datasets/dir_name/' --result_path='./datasets/dir_name/result/'

## **Generate Semantic label**

생성된 binary mask를 이용하여 semantic label을 생성합니다.

    cd utils
    python generate_semantic_label_contour.py 


## **Convert TFDS dataset**

생성된 Semantic label을 tf.data format으로 변환하기 위해 tensorflow datasets 라이브러리를 사용합니다.<br>

증강이 적용된 RGB 이미지와 semantic label이 저장된 이미지를 다음과 같은 폴더로 이동시킵니다.


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
<hr/>

# 4. Train

Binary segmentation/ Semantic segmentation에 대한 학습 코드는 각 스크립트별로 구분되어 있습니다.

학습하기전 tf.data의 메모리 할당 문제로 인해 TCMalloc을 사용하여 메모리 누수를 방지합니다.

    1. sudo apt-get install libtcmalloc-minimal4
    2. dpkg -L libtcmalloc-minimal4

    2번을 통해 설치된 TCMalloc의 경로를 저장합니다

## Training binary segmentation

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python binary_train.py

## Training semantic segmentation

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python semantic_train.py

    '-h' 를 통해 학습에 필요한 args를 설정할 수 있습니다.
    python3 semantic_train.py -h
    usage: semantic_train.py [-h] [--model_prefix MODEL_PREFIX] [--batch_size BATCH_SIZE] [--epoch EPOCH] [--lr LR] [--weight_decay WEIGHT_DECAY] [--optimizer OPTIMIZER] [--model_name MODEL_NAME]
                            [--dataset_dir DATASET_DIR] [--checkpoint_dir CHECKPOINT_DIR] [--tensorboard_dir TENSORBOARD_DIR] [--use_weightDecay USE_WEIGHTDECAY] [--load_weight LOAD_WEIGHT]
                            [--mixed_precision MIXED_PRECISION] [--distribution_mode DISTRIBUTION_MODE]

<br>
<hr>

# 5. Predict
Training 이후 모델의 추론 결과를 테스트해볼 수 있습니다.

## Predict binary segmentation

    python binary_predict.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

## Predict semantic segmentation
    python semantic_predict.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

<br>
<hr>

# Inference real-time
학습된 가중치를 이용하여 카메라를 이용하여 실제 추론을 테스트해볼 수 있습니다.

### **1. Inference with PyRealSense Camera <br>**
&nbsp; &nbsp; &nbsp; Intel RealSense Camera를 직접 사용하는 경우
### **2. Inference with RealSense-ROS <br>**
&nbsp; &nbsp; &nbsp; Intel RealSense Camera를 ROS(Robot Operating System)로 이미지 데이터를 subcribe하여 사용하는 경우

<br>
<br>

## 1. Inference with PyRealSense Camera
    1. 연결된 RealSense 카메라의 시리얼 넘버를 확인합니다.
        terminal
            (park) park@park:~$ rs-enumerate-devices 
            Device info: 
                Name                          : 	Intel RealSense L515
>                Serial Number                 : 	f0350818
                Firmware Version              : 	01.05.08.01
                Recommended Firmware Version  : 	01.05.08.01
                Physical Port                 : 	4-3.1.1-48
                Debug Op Code                 : 	15
                Product Id                    : 	0B64
                Camera Locked                 : 	YES
                Usb Type Descriptor           : 	3.2
                Product Line                  : 	L500
                Asic Serial Number            : 	0003b661b825
                Firmware Update Id            : 	0003b661b825

    2. 
        python semantic_realtime_full.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5', --serial_num='f0350818'

## 2. Inference with RealSense-ROS
ROS 연동 관련하여 사전 지식 및 세팅은 개별적으로 진행해야 합니다.
테스트 환경은 ROS-melodic입니다.


    ROS
        1.roslaunch realsense2_camera rs_camera.launch
        2.rqt
        3.Check your camera id
        4. Modify camera id values (camera_infos.json)
    
    PYTHON
        1. python3 inference_robot_full.py


# Reference
<hr>

1. [DDRNet : https://github.com/ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet)
2. [CSNEt-seg https://github.com/chansoopark98/CSNet-seg](https://github.com/chansoopark98/CSNet-seg)