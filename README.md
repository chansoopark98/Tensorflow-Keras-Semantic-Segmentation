<h1>ESDet</h1>

> Hole detection : Bracket segmentation

이 저장소는 **Hole Detection**을 위해 **Binary segmentation, Semantic segmentation**을 응용하여 구현하였습니다.  

<br>

### **Keyword:** Salient Object Detection, Binary Segmentation, Semantic Segmentation, Deep learning, Lightweight, distribute training

<br>

### **Use library:** Tensorflow, Keras, OpenCV, ROS
### **Options:** Distribute training, Custom Data
### **Models:** DDRNet-23-Slim 


<br>
<hr/>

## Table of Contents

 1. [Overview of Model](#Overview of Model)
 2. [Preferences](#Preferences)
 3. [Preparing datasets](#Preparing-datasets)
 4. [Train](#Train)
 5. [Eval](#Eval)
 6. [Predict](#Predict)

<br>
<hr/>

# 1. Overview of Model

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

$$ J(w)=−1N∑i=1N[yilog(y^i)+(1−yi)log(1−y^i)] $$

#### $w$ &nbsp;  refer to the model parameters, e.g. weights of the neural network <br>
#### $yi$ &nbsp; is the true label <br>
#### $\hat{y_i}$ &nbsp; is the predicted label

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

    ### $L(y,p^)=−αy(1−p^)γlog(p^)−(1−y)p^γlog(1−p^)$
    
    where

    y∈{0,1} is a binary class label, <br>

    p^∈[0,1] is an estimate of the probability of the positive class, <br>

    γ is the focusing parameter that specifies how much higher-confidence correct predictions contribute to the overall loss (the higher the γ, the higher the rate at which easy-to-classify examples are down-weighted). <br><br>

    α is a hyperparameter that governs the trade-off between precision and recall by weighting errors for the positive class up or down (α=1 is the default, which is the same as no weighting),

    <br>
    
<br>
<hr/>

# 2. Preferences

ESDet은 Tensorflow 기반 코드로 작성되었습니다. 코드는 **Windows** 및 **Linux(Ubuntu)** 환경에서 모두 동작합니다.
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
        ROS
        </td>
        <td>
        Melodic
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
        NVIDIA RTX3090 24GB
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

Bracket image labels(binary and semantic label)이 필요한 경우엔 개인 이메일 chansoo0710@gmail.com으로 코멘트를 남겨주세요
<br>

## **Custom dataset labeling process**
* Binary mask label
    1. 데이터 생성 및 마스크 레이블링
    2. 데이터 증강
        * 이미지 시프트
        * 이미지 블러링
        * 이미지 회전
        * 이미지 배경 합성
* Semantic label
    1. 데이터 생성 및 시멘틱 레이블링
    2. 데이터 증강
        * 이미지 시프트
        * 이미지 블러링
        * 이미지 회전
        * 이미지 배경 합성 (랜덤 이미지 크기 조정)


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
            (park) park@park-plaif:~$ rs-enumerate-devices 
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



<!-- ![001070 jpg](https://user-images.githubusercontent.com/60956651/110722231-49632f00-8255-11eb-9351-165d9efac7c2.jpg)
![002107 jpg](https://user-images.githubusercontent.com/60956651/110722280-54b65a80-8255-11eb-8005-0ddd88f33082.jpg)   -->

# Reference
<hr>

1. [DDRNet : https://github.com/ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet)
2. [CSNEt-seg https://github.com/chansoopark98/CSNet-seg](https://github.com/chansoopark98/CSNet-seg)