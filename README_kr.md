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

# Reference
<hr>

1. [DDRNet : https://github.com/ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet)
2. [CSNEt-seg https://github.com/chansoopark98/CSNet-seg](https://github.com/chansoopark98/CSNet-seg)