<h1>End-to-End Semantic Segmentation</h1>

> All about Tensorflow/Keras semantic segmentation


## Tensorflow/Keras based semantic segmentation repository  

<br>

 [![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/total.svg)]() 

 [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FTensorflow-Keras-Realtime-Segmentation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23C41010&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)



<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
 <img src="https://img.shields.io/github/forks/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
 <img src="https://img.shields.io/github/stars/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
 <img src="https://img.shields.io/github/license/chansoopark98/Tensorflow-Keras-Realtime-Segmentation">
 </p>

<br>


<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Tensorflow-FF6F00.svg?&style=for-the-badge&logo=Tensorflow&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <br>
 <br>
</p>

- ## 한국어 [README.md](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/README_kr.md) 지원


<br>


<p align="center">

 ![main_image_1](https://user-images.githubusercontent.com/60956651/181407216-63498ca5-7668-4188-853b-c48506534b9e.png)

</p>

<div align=center>
    Cityscapes Image segmentation results (with ignore index)
</div>

<br>

<p align="center">

![166](https://user-images.githubusercontent.com/60956651/181407706-1d2ba5cd-fe9f-419f-aa03-e44e6e77a40e.png)

</p>

<div align=center>
    Cityscapes Image segmentation result (without ignore index)
</div>

<br>


### Supported options
- Data preprocessing
- Train
- Evaluate
- Predict real-time
- TensorRT Converting
- Tensorflow docker serving

<br>

### **Use library** 
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

The dependencies of this repository are:

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
        3.8.13~
        </td>
    </tr>
    <tr>
        <td>
        CUDA
        </td>
        <td>
        11.1~
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

Download the package from the **Anaconda (miniconda)** virtual environment for training and evaluation.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<br>
<hr/>

# 3. Preparing datasets

The **Dataset** required by the program uses the **Tensorflow Datasets** library [TFDS](https://www.tensorflow.org/datasets/catalog/overview).

<br>

## **Custom dataset labeling process**
Custom data image labeling was done using a tool called **CVAT** (https://github.com/openvinotoolkit/cvat).

After the labeling operation is completed, the export format of the dataset is created in **CVAT** as **Segmentation mask 1.1 format**.

You can check the RGB values for each class in **labelmap.txt** of the created dataset.


* **How to Use?**
    1. Label semantic data (mask) using CVAT tool
    2. Raw data augmentation
        * Image shift
        * Image blurring
        * Image rotate
        * Mask area image conversion..etc

<br>
First, for images without a foreground, CVAT does not automatically create a label.
Assuming there is no foreground object as shown below, a zero label is created.
<br>

<br>

    cd data_augmentation
    python make_blank_label.py

<br>

Second, 

    python augment_data.py
    
Perform augmentation by specifying the path to the options below.

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

You can choose which options to augment directly in __main__: at the bottom of the code. Modify this section to suit your preferred augmentation method.

<br>

## **Convert TFDS dataset**

We use the tensorflow datasets library to convert the generated semantic labels into tf.data format.

<br>

Move the augmented RGB image and semantic label saved image to the folder below.

<br>


    └── dataset 
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

Compress that directory to 'full_semantic.zip'.

    zip full_semantic.zip ./*

When the compression is complete, it should be set like the corresponding path.

    
    └──full_semantic.zip
        ├── rgb/  # RGB image.
        |   ├── image_1.png 
        |   └── image_2.png
        └── gt/  # Semantic label.    
            ├── image_1_mask.png 
            └── image_2_output.png

Then, move full_semantic.zip after creating a folder structure like the one below.

    /home/$USER/tensorflow_datasets/downloads/manual

Finally, build the dataset.
    
    cd hole-detection/full_semantic/
    tfds build

    # if build is successfully
    cd -r home/$USER/tensorflow_datasets/
    cp full_semantic home/$USER/hole-detection/datasets/
<br>

### **Caution!**

The work output of **[augment_data.py](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/data_augmentation/augment_data.py)** basically consists of three paths: RGB, MASK, and VIS_MASK.<br>

**VIS_MASK** is not a label to be actually used, it is for visual confirmation, so do not use it in the work below. <br>

<br>

<hr/>

# 4. Train

Because of memory allocation issues in tf.data before training, use TCMalloc to avoid memory leaks.

    1. sudo apt-get install libtcmalloc-minimal4
    2. dpkg -L libtcmalloc-minimal4

    !! Remember the path of TCMalloc installed through #2

<br>

## Training semantic segmentation

<br>

**How to RUN?**
    
### When use **Single gpu**

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

### When use **Multi gpu**

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py --multi_gpu


## **Caution!**
This repository supports training and inference in single-GPU and multi-GPU environments. <br>
When using Single-GPU, you can set the GPU number and use it. <br>
Take a look at **[train.py](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/train.py)** --help and add the setting value required for training as an argument value. <br>

<br>
<hr>

# 5. Eval
Evaluate the accuracy of the model after training and compute the inference rate. <br>
<br>
Calculation items: FLOPs, MIoU metric, Average inference time
<br>

    python eval.py --checkpoint_dir='./checkpoints/' --weight_name='weight.h5'

<br>
If you want to check the inference result, add the --visualize argument.



<hr>

# 6. Predict
Web-camera or stored video can be inferred in real time. <br>
<br>

**When video realtime inference**

    python predict_video.py

<br>

**Web-cam realtime inference**

    python predict_realtime.py


<br>

If you want to check the inference result, add the **--visualize** argument.

<br>
<hr>

# 7. Convert TF-TRT
Provides TF-TRT conversion function to enable high-speed inference.
Install tensorRT before conversion.


## 7.1 Install CUDA, CuDNN, TensorRT files

<br>

The CUDA and CuDNN and TensorRT versions used based on the currently written code are as follows. <br>
Click to go to the install link. <br>
Skip if CUDA and CuDNN have been previously installed.

<br>

### CUDA : **[CUDA 11.1](https://www.tensorflow.org/datasets/catalog/overview)**
### CuDNN : **[CuDNN 8.1.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz)**
### TensorRT : **[TensorRT 7.2.2.3](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.2/tars/tensorrt-7.2.2.3.ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz)**

<br>

## 7.2 Install TensorRT
<br>

Activate the virtual environment. (If you do not use a virtual environment like Anaconda, omit it)
    
    conda activate ${env_name}

<br>

Go to the directory where you installed TensorRT, unzip it and upgrade pip.

    tar -xvzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
    pip3 install --upgrade pip

Access the bash shell using an editor and add environment variables.

    sudo gedit ~/.bashrc
    export PATH="/usr/local/cuda-11.1/bin:$PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/park/TensorRT-7.2.2.3/onnx_graphsurgeon
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/park/TensorRT-7.2.2.3/lib"

Install the TensorRT Python package.

    cd python
    python3 -m pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl

    cd ../uff/
    python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

    cd ../graphsurgeon
    python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

    cd ../onnx_graphsurgeon
    python3 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl

Open Terminal and check if the installation was successful.

![test_python](https://user-images.githubusercontent.com/60956651/181165197-6a95119e-ea12-492b-9587-a0c5badc73be.png)

<br>

## 7.3 Convert to TF-TensorRT

Pre-trained **graph model (.pb)** is required prior to TF-TRT transformation. <br>
If you do not have a graph model, follow the procedure **7.3.1**, if you do, skip to **7.3.2**.


- ### 7.3.1 If there is no graph model

    In this repository, if there are weights trained through **[train.py](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/train.py)**, it provides a function to convert it to a graph model.

    Enable graph saving mode with **--saved_model** argument in **[train.py](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/train.py)**. And it adds the path where the weights of the trained model are stored.

        python train.py --saved_model --saved_model_path='your_model_weights.h5'

    The default saving path of the converted graph model is **'./checkpoints/export_path/1'** .

    ![saved_model_path](https://user-images.githubusercontent.com/60956651/181168185-376880d3-b9b8-4ea7-8c76-be8b498e34b1.png)

    <br>

- ### 7.3.2 Converting

    If the **(.pb)** file exists, run the script below to perform the conversion.

        python convert_to_tensorRT.py ...(argparse options)

    Converting the model via the TensorRT engine. The engine is built based on a fixed input size, so check the **--help** argument before running the script.

    <br>
    
    The options below are provided. <br>

    **Model input resolution** (--image_size), **.pb file directory path** (input_saved_model_dir) <br>
    
    **TensorRT converting model save path** (output_saved_model_dir), **Set converting floating point mode** (floating_mode)

    <br>

<hr>

# 8. Tensorflow serving

Provides the ability to serve pre-trained graph models (.pb) or models built with the TensorRT engine.

<br>

Tensorflow serving is a tool that provides inference services within a Docker virtual environment.

Before working, install Docker for the current operating system version. (https://docs.docker.com/engine/install/ubuntu/) <br>


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

If you have Docker installation and model file ready, you can run it right away. Before running, look at the options to configure the Serving server.

    docker run 
	--runtime=nvidia # Settings to use nvidia-gpu in docker
	-t # Use tty
	-p 8500:8500 # Port address to open in docker environment
	--rm # Automatically delete docker containers when not in use
	-v "model_path:/models/test_model2" {1}:{2} -> {1} is the path where the .pb file is located. {2} is the path where the model will be deployed in Docker (request request using the name test_model2)
	-e MODEL_NAME=test_model2 # gRPC, model name to be called in REST API
	-e NVIDIA_VISIBLE_DEVICES="0" # gpu number to use
	-e LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 # Specifying cuda-11.1 environment variables for building TensorRT engine (TensorRT engine 7.2.2.3)
	-e TF_TENSORRT_VERSION=7.2.2 tensorflow/serving:2.6.2-gpu # Set the TensorRT version and install the tensorflow-gpu version for that version
	--port=8500 # Port number to use when serving (must be the same as Docker port setting)


Additional information can be found with the **--help** argument at the end of the command. <br>

Please refer to **[tf_serving_sample.py](https://github.com/chansoopark98/Tensorflow-Keras-Realtime-Segmentation/blob/main/tf_serving_sample.py)** for an example of accessing the Tensorflow-serving server and making an inference request. <br>


# References
<hr>

1. [DDRNet : https://github.com/ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet)
2. [CSNet-seg : https://github.com/chansoopark98/CSNet-seg](https://github.com/chansoopark98/CSNet-seg)
3. [A study on lightweight networks for efficient object detection based on deep learning](https://doi.org/10.14400/JDC.2021.19.5.307)
4. [Efficient shot detector](https://www.mdpi.com/2076-3417/11/18/8692)