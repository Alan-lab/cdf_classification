import torch
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据。"""
    return 4

def load_valid_data(batch_size, resize=None):  
    # 下载/加载评估数据集，检验模型的实际效果。
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize([resize, resize]))
    trans = transforms.Compose(trans)

    valid_path = "/home/lfq/data/catsdogsflowers/valid/"
    # 使用torchvision.datasets.ImageFolder读取数据集指定的valid文件夹
    # shuffle=True表示随机读取
    valid_data = torchvision.datasets.ImageFolder(valid_path, transform=trans)
    return (data.DataLoader(valid_data, batch_size,
                            shuffle=True, num_workers=get_dataloader_workers()))

def accuracy(y_hat, y, type_num):
    #Compute the number of correct predictions.
    #y_hat为softmax预测的类别向量，y为真实的类别向量，type为预测种类
    #Defined in :numref:`sec_softmax_scratch
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)

    #定义数组，偶数索引存储预测正确的图片数，奇数索引存储一个batch下的该种类的总图片数
    cor = torch.zeros(type_num * 2)

    for i in range(type_num):
        #计算对第i类图片的预测正确数
        for j in range(len(y)):
            if(y[j] == i):
                cmp = y_hat[j].type(y.dtype) == y[j]
                if(i == 0):
                    cor[i+1] += 1
                else:
                    cor[i*2+1] += 1
                cor[i*2] += float(d2l.astype(cmp, y.dtype))
    # cor = cor.to(device)
    return cor


def predict(model, valid_iter, type, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    print('training on', device)
    timer, num_batches = d2l.Timer(), len(valid_iter)
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(6)
    for i, (X, y) in enumerate(valid_iter):
        timer.start()
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        with torch.no_grad():
            batch_acc = accuracy(y_hat, y, type)
            metric.add(batch_acc[0], batch_acc[1], batch_acc[2],
                       batch_acc[3], batch_acc[4], batch_acc[5])
    timer.stop()
    valid_acc_0 = metric[0] / metric[1]
    valid_acc_1 = metric[2] / metric[3]
    valid_acc_2 = metric[4] / metric[5]
    print(f'valid cat acc {valid_acc_0:.3f}, valid dog acc {valid_acc_1:.3f}',
          f'valid flower acc {valid_acc_2:.3f}')

#type为预测的分类数, 本任务中为三类，cat,dog,flower
batch_size, type =256, 3
model = torch.load('/home/lfq/workspace/classification/model.pkl')
valid_iter = load_valid_data(batch_size, resize=96)
predict(model, valid_iter, type, d2l.try_gpu())



