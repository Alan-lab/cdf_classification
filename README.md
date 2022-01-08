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


3.Predict

Prepare your valid dataset like this (https://github.com/Alan-lab/data/catsdogsflowers/valid1).
```bash
python Predict/predict.py
```


4.Class Activation Map
The response size of the feature map is mapped to the original image, allowing readers to understand the effect of the model more intuitively.
Prepare your picture like this (https://github.com/Alan-lab/data/Dogcatflower/test/flower/flower.1501.jpg).
```bash
python Viewer/Grad_CAM.py
```

5. More details can be found in [folder](https://github.com/Alan-lab/cdf_classification).

## The Experimental Result
Few results are shown as follows.

1. Preformance

| dataset | Cat-acc | Dog-acc | flower-acc |
| :---: | :---: | :---: | :---: |
| Dogcatflower_2_train | 96.2 | 88.7 | 93.6 |
| Dogcatflower_2_test | 72.7 | 69.2 | 89.7 |
| catsdogsflowers_valid1 | 75.1 | 76.9 | 91.4 |
| catsdogsflowers_valid2 | 75.5 | 73.5 | 92.9 |

2.Visualization

***Postive sample***
![fig1](https://user-images.githubusercontent.com/54443297/148637872-d17d3438-239c-49b7-9ad8-61bb8e96cce9.png)
![fig2](https://user-images.githubusercontent.com/54443297/148637879-8e6861ce-12ff-48dd-83e4-5748acacff09.png)
![fig3](https://user-images.githubusercontent.com/54443297/148637655-11019508-ceab-481e-9491-bb0b95002c4e.png)

***Negative sample***
![fig4](https://user-images.githubusercontent.com/54443297/148638009-77de2573-8379-43e1-bb37-82d7ac598cf1.png)

***Multi-attention***
![show_attention](https://user-images.githubusercontent.com/54443297/148638388-3b087944-3b15-41bb-9a11-73caae9fbcbd.jpg)

## Acknowledgments
This work is mainly supported by (https://courses.d2l.ai/zh-v2/) and CSDN.

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Lailanqing ([lailanqing_charlie@163.com](lailanqing_charlie@163.com)).

