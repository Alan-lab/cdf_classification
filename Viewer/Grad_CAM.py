import cv2
import os
import numpy as np
import torch


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    C, H, W = img.shape

    #self-attention层的输出为256*36
    feature_map = np.reshape(feature_map, (256, 6, 6))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    #将img恢复成原始尺寸
    img = img.T
    cam_img = 0.3 * heatmap + 0.7 * img

    #cam_img = cam_img.reshape(96, 96, 3)
    path_cam_img = os.path.join(out_dir, "heat_map.jpg")
    cv2.imwrite(path_cam_img, cam_img)
    path_cam_img = os.path.join(out_dir, "raw.jpg")
    cv2.imwrite(path_cam_img, img)


if __name__ == '__main__':
    path_img = "/home/lfq/data/Dogcatflower/test/flower/flower.1501.jpg"
    output_dir = '/home/lfq/workspace/classification/result'

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 读取数据集的类别标签
    classes = {0: 'cat', 1: 'dog', 2: 'flower'}

    # 读取 imagenet数据集的某类图片
    image = cv2.imread(path_img)

    # 图片预处理成data_x
    image = cv2.resize(image, (96, 96))
    image = image.T

    arr = np.asarray(image, dtype="float32")

    data_x = np.empty((1, 3, 96, 96), dtype="float32")

    data_x[0, :, :, :] = arr
    data_x = data_x / 255
    data_x = torch.from_numpy(data_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_x = data_x.to(device)
    img_input = data_x

    # 加载预训练模型

    # net = models.squeezenet1_1(pretrained=False)
    pthfile = '/home/lfq/workspace/classification/model.pkl'
    net = torch.load(pthfile)
    #net.load_state_dict(torch.load(pthfile))
    net.eval()  # 使用eval()属性
    print(net)

    # 注册hook
    net._modules.get('5').register_forward_hook(farward_hook)  # 9
    net._modules.get('5').register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(image, fmap, grads_val, output_dir)

