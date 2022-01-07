import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import torch
from torch.nn import functional as F


def test_graph():
    plt.figure()
    #train loss的折线图分布
    #plt.plot(X, train_l, label="train loss", linestyle=":")

    X=range(10)
    train_acc=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    test_acc=[0.3,0.1,0.2,0.2,0.5,0.6,0.8,0.5,0.6,1]
    # train acc的折线图分布
    plt.plot(X, train_acc,label="train acc", linestyle="--")

    # test acc的折线图分布
    plt.plot(X, test_acc, label="test acc", linestyle="-.")
    plt.legend()
    #plt.title("")
    #plt.xlabel(str(ALL_X[0]))
    plt.ylabel("%")
    plt.show()
    path = '/home/lfq/workspace/classification/result'
    save_path = os.path.join(os.path.abspath(path), 'find_opt_par' + '.jpg')
    plt.savefig(save_path)

def test_predict():
    # 读取数据集的类别标签
    classes = {0: 'cat', 1: 'dog', 2: 'flower'}

    #下载训练好的模型及图片
    model1 = torch.load('/home/lfq/workspace/classification/model.pkl')
    image = cv2.imread("/home/lfq/data/catsdogsflowers/valid/flower/126012913_edf771c564_n.jpg")

    # 图片预处理成模型要求的输入大小
    image = cv2.resize(image, (96, 96))
    image = image.T

    arr = np.asarray(image, dtype="float32")

    data_x = np.empty((1, 3, 96, 96), dtype="float32")

    data_x[0, :, :, :] = arr
    data_x = data_x / 255
    data_x = torch.from_numpy(data_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_image = data_x.to(device)

    #将预处理好的图片输入到训练好的模型中
    logit = model1(in_image)
    h_x = F.softmax(logit, dim=1).data.squeeze()

    print(h_x.shape)  # torch.Size([3]), 模型的输出
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()  # 概率值排序
    idx = idx.cpu().numpy()  # 类别索引排序，概率值越高，索引越靠前

    # 取概率值为前3的类别,观察类别名和概率值
    print('The probability of each category as follows:')
    for i in range(0, 3):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    '''
    0.996 -> flower
    0.004 -> dog
    0.000 -> cat
    '''
    idx = np.argmax(logit.cpu().data.numpy())
    print("Predict result: {}".format(classes[idx]))

test_predict()