# cdf_att_classification
classes = {0: 'cat', 1: 'dog', 2: 'flower'}

In this project we use both Resnet and Self-attention layer for cdf-Classification.
Specifically, For Resnet, we extract low level features from Convolutional Neural Network (CNN) trained on Dogcatflower_2 dataset(details show later).  
We take inspiration from the Self-attention mechanism which is a prominent method in cv domain. 
We also use Grad-CAM algorithm to Visualize the gradient of the back propagation of the pretrain model to understand this network.
The code is released for academic research use only. For commercial use, please contact [lailanqing_charlie@163.com].

## Installation

Clone this repo.
```bash
git clone https://github.com/Alan-lab/cdf_classification
cd cdf_classification/
```

This code requires pytorch, python3.7, cv2, d2l. Please install it.


## Dataset Preparation
For cdf_classification, the datasets must be downloaded beforehand. Please download them on the respective webpages.  Please cite them if you use the data. 

**Preparing Cat and Dog Dataset**. The dataset can be downloaded [here](https://www.kaggle.com/tongpython/cat-and-dog). 

**Preparing flower Dataset**. The dataset can be downloaded [here](https://www.kaggle.com/alxmamaev/flowers-recognition). 

You can also download Dogcatflower_2 dataset(made from above datasets) use the following link:

Link:https://pan.baidu.com/s/1ZcP_isbbRQBq9BHU6p_VtQ  

key:oz7z 


## Training New Models
New models can be trained with the following commands.

1. Prepare your own dataset like this (https://github.com/Alan-lab/data/Dogcatflower_2).

2. Training:
```bash
python main.py
```
model.pth will be extrated in the folder `./cdf_classification`. 

**If av_test_acc < 0.75, model.pth will not save(d2l.train_ch6).**


3.predict
Prepare your valid dataset like this (https://github.com/Alan-lab/data/catsdogsflowers/valid).
```bash
python Predict/predict.py
```

#The result of trained model in valid dataset
```bash
valid cat acc 0.755, valid dog acc 0.735 valid flower acc 0.929
```

4.Class Activation Map
The response size of the feature map is mapped to the original image, allowing readers to understand the effect of the model more intuitively.
Prepare your picture like this (https://github.com/Alan-lab/data/Dogcatflower/test/flower/flower.1501.jpg).
```bash
python Viewer/Grad_CAM.py
```

5. More details can be found in [folder](https://github.com/Alan-lab/cdf_classification).

## Acknowledgments
This work is mainly supported by (https://courses.d2l.ai/zh-v2/) and CSDN.

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Lailanqing ([lailanqing_charlie@163.com](lailanqing_charlie@163.com)).
