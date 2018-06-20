# Steganography Recognition using Deep Learning

## Overview

Steganography is the practice of concealing a file, message, image, or video within another file, message, image, or video.
This code repository contains scipts that encrypts and decrypts text into/from PNG images using bitwise operators (on a chosen bit layer)
We produced out own Dataset, using the cifar-10 image DataSet, and running the encryption Script supplied in this repository

## Dependencies 
We used Python 3.6, the required libraries are:
- PIL (Python Image Library) 
- Numpy
- Pytorch and TorchVision
- Matplotlib

## Dataset
We created our own Dataset, wev'e used the CIFAR-10 Dataset (can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html) 

To extract the actual PNG images wev'e used this code repository: https://gist.github.com/MatthiasWinkelmann/fbca5503c391c265f958f49b60af6bae

Once the image are extracted use the steganography_encode.py script to inject the messages into required % of images (we used 50% but can be changed.)
The BitLayer, jumps between bits, and channel can be chosen randomly, but we've used the 3rd layer, with jump of 2 pixels and the red channel for encryption.

Another option is to use bigger images, you can use Google Image Crawler (https://github.com/hellock/icrawler) insted of using CIFAR-10 (Make sure all images are same size, and PNG format).
Orgnize the Data in 2 different folders (According to prediction labels - already being orgenized as part of encoder output), one which contains the Steganography Images, and one with the regular ones.

To use the Steganography script:
Choose the % of Steganography images (set to 50%):
```
PROB = 50
```

Choose Bit layer, jumps and channel on second line or remove second line and remove # from first line to use random selection
```
# jump, bl, channel = random_bl_and_jump(img, m_len)
jump, bl, channel = 2, 3, 1
```

then enter your Image folder path and required output folder in steganography/steganography_encode.py
```
# where to get input image from
dir = "./raw train png/"
# output dir
outdir = 'ready train'
```

## Recognition Model
We used a Neaural Network with 2 hidden layers, the input size for the CIFAR-10 is 32*32 (1024 neurons) pixels, 50 neurons on the 1st hidden and 20 on the 2nd hidden layer
the output is binary - does the image contains encrypted data or not.


## Train your model
on the first lines on NN.py change your model Hyper Parameters we used LR=0.01, Batch size of 10, 20 Epochs and image size we used 32x32x3(RGB) CIFAR-10 images.
also define your Dataset directory
```
LR = 0.01
BATCH_SIZE = 10
NUM_OF_EPOCHS = 20
IMAGE_SIZE = 32 * 32 * 3
DATASET_DIR = '/home/daniel/Documents/stang_proj/stenography/ready train'
```

Once all parameters defiend you are good to go, running the script will train the model and output results

##Results



## TEAM
[Daniel Yaakov](https://github.com/danielYaakov) 

[Shlomi Amichay](https://github.com/ShlomiAmichay)
