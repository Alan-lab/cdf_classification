import matplotlib.pyplot as plt
import os
from d2l import torch as d2l
from IPython import display

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    #d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def draw(ALL_X, train_l, train_acc, test_acc):
    # 绘制随变量X(如batch,epoch)的训练精度变化曲线
    # X = ALL_X[1:len(ALL_X)]
    # plt.figure()
    # train loss的折线图分布
    # plt.plot(X, train_l, label="train loss", linestyle=":")
    # train acc的折线图分布
    # plt.plot(X, train_acc,label="train acc", linestyle="--")

    # test acc的折线图分布
    # plt.plot(X, test_acc, label="test acc", linestyle="-.")
    # plt.legend()
    # plt.title("")
    # plt.xlabel(str(ALL_X[0]))
    # plt.ylabel("%")
    # plt.show()

    path = '/home/lfq/workspace/classification/result'
    save_path = os.path.join(os.path.abspath(path), str(ALL_X[0]) + '.jpg')
    plt.savefig(save_path)