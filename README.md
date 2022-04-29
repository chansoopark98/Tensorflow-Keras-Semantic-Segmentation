<h1>Tensorflow Keras Realtime semantic segmentation</h1>

<h2>with Custom data, Cityscapes, RGB-D, NYU-DepthV2</h2>

> Binary/Semantic segmentation with Custom data

This repository has implemented everything from data labeling to real-time inference for segmentation using **custom datasets**.

 &nbsp; **(Binary segmentation, Semantic segmentation)**

<br>

### **Keyword:** Salient Object Detection, Binary Segmentation, Semantic Segmentation, Deep learning, Lightweight, distribute training

<br>

### **Use library:** Tensorflow, Keras, OpenCV, ROS
### **Options:** Distribute training, Custom Data
### **Models:** DDRNet-23-Slim 


<br>

### 한국어 [ReadME.md](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/README_kr.md)도 지원합니다.
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

Download the package from the **Anaconda (miniconda)** virtual environment for training and evaluation.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<br>
<hr/>

# 3. Preparing datasets

The **Dataset** required by the program uses the **Tensorflow Datasets library**  ([TFDS](https://www.tensorflow.org/datasets/catalog/overview)).

<br>

## **Custom dataset labeling process**
* Binary mask label
    1. Data Generation and Mask Labeling
    2. Data Augmentation
         * Image shift
         * Image blurring
         * Rotate image
         * Image background synthesis
* Semantic label
    1. Data Generation and Semantic Labeling
    2. Data Augmentation
         * Image shift
         * Image blurring
         * Rotate image
         * Image background compositing (random image resizing)


## **Generate Binary mask label**


Generate it by running utils/generate_binary_mask.py.

    cd utils
    python generate_binary_mask.py --image_path='./datasets/dir_name/' --result_path='./datasets/dir_name/result/'

## **Generate Semantic label**

A semantic label is created using the generated binary mask.

    cd utils
    python generate_semantic_label_contour.py 


## **Convert TFDS dataset**

We use the tensorflow datasets library to convert the generated semantic labels into tf.data format.<br>

Move the RGB image with augmentation applied and the image with semantic label saved to the following folder.


    └── dataset 
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

Compress that directory into 'full_semantic.zip'.

    zip full_semantic.zip ./*

After the compression is complete, it should be set like the corresponding path.

    
    └──full_semantic.zip
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

Then, move full_semantic.zip after creating the following folder structure.

    /home/$USER/tensorflow_datasets/downloads/manual

Finally, build the dataset.
    
    cd hole-detection/full_semantic/
    tfds build

    # if build is successfully
    cd -r home/$USER/tensorflow_datasets/
    cp full_semantic home/$USER/hole-detection/datasets/
<br>
<hr/>

# 4. Train

The training code for binary segmentation/semantic segmentation is separated for each script.

Because of memory allocation issues in tf.data before training, use TCMalloc to avoid memory leaks.

    1. sudo apt-get install libtcmalloc-minimal4
    2. dpkg -L libtcmalloc-minimal4

    Save the path of TCMalloc installed through '2'

## Training binary segmentation

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python binary_train.py

## Training semantic segmentation

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python semantic_train.py

    You can set the args required for training through '-h' .
    python3 semantic_train.py -h
    usage: semantic_train.py [-h] [--model_prefix MODEL_PREFIX] [--batch_size BATCH_SIZE] [--epoch EPOCH] [--lr LR] [--weight_decay WEIGHT_DECAY] [--optimizer OPTIMIZER] [--model_name MODEL_NAME]
                            [--dataset_dir DATASET_DIR] [--checkpoint_dir CHECKPOINT_DIR] [--tensorboard_dir TENSORBOARD_DIR] [--use_weightDecay USE_WEIGHTDECAY] [--load_weight LOAD_WEIGHT]
                            [--mixed_precision MIXED_PRECISION] [--distribution_mode DISTRIBUTION_MODE]

<br>
<hr>

# 5. Predict
After training, you can test the inference results of your model.

## Predict binary segmentation

    python binary_predict.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

## Predict semantic segmentation
    python semantic_predict.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

<br>
<hr>

# Inference real-time
You can test real-time inference with the camera using the learned weights.

### **1. Inference with PyRealSense Camera <br>**
&nbsp; &nbsp; &nbsp; When using the Intel RealSense Camera directly
### **2. Inference with RealSense-ROS <br>**
&nbsp; &nbsp; &nbsp; When using Intel RealSense Camera by subcribe image data with ROS(Robot Operating System)

<br>
<br>

## 1. Inference with PyRealSense Camera
    1. Check the serial number of the connected RealSense camera.
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
In relation to ROS interworking, prior knowledge and setting must be carried out individually.
The test environment is ROS-melodic.


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
