from Pre_process.load_data import load_data
from d2l import torch as d2l
from Model.model_test import net
from Viewer.draw_graph import draw

def train():
    lr, num_epochs, batch_size = 0.01, 96, 256

    #定义自注意力层数，自注意力头
    num_layers, num_heads = 1, 3

    train_iter, test_iter = load_data(batch_size, resize=96)
    result = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


def opt_par(lr, num_epochs, batch_size):
    #寻找最优的epoch和batch_size
    X = ['epoch']
    train_l, train_acc, test_acc=[], [], []
    train_iter, test_iter = load_data(batch_size, resize=96)
    for num_epochs in range(1,41,1):
       result = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
       train_l = train_l + [result[0]]
       train_acc = train_acc + [result[1]]
       test_acc = test_acc + [result[2]]
       X = X + [num_epochs]
    draw(X,train_l,train_acc, test_acc)
    X = ['batch_size']
    train_l, train_acc, test_acc=[], [], []
    for batch_size in range(16,256,16):
        train_iter, test_iter = load_data(batch_size, resize=96)
        result = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
        train_l = train_l + [result[0]]
        train_acc = train_acc + [result[1]]
        test_acc = test_acc + [result[2]]
        X = X + [batch_size]
    draw(X,train_l,train_acc, test_acc)
