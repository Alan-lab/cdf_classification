import torch
import draw_graph as draw
import os

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(9, 16),
                  cmap='Reds'):
    """显示矩阵热图"""
    draw.use_svg_display()

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = draw.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    path = '/home/lfq/workspace/classification/result'
    save_path = os.path.join(os.path.abspath(path), 'show_attention' + '.jpg')
    draw.plt.savefig(save_path)

num_layers, num_heads, num_steps = 1, 3, 12

#下载测试例中对应的self-attention层中, 各个自注意力头得到的权重
attention_weights = torch.load('/home/lfq/workspace/classification/attention_value.csv')
attention_weights = attention_weights.reshape((num_layers, num_heads,
    -1, num_steps))
attention_weights = attention_weights.cpu()
print(attention_weights.shape)
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries', titles=['Head %d' % i for i in range(1, 4)])
