import torchvision
from torch.utils import data
from torchvision import transforms

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据。"""
    return 4


def load_data(batch_size, resize=None):  #@save
     #下载/加载数据集，然后将其加载到内存中。
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize([resize,resize]))
    trans = transforms.Compose(trans)

    path = "/home/lfq/data/Dogcatflower_2"
    train_path=path+"/train"
    test_path=path+"/test"
    #使用torchvision.datasets.ImageFolder读取数据集指定train和test文件夹
    train_data = torchvision.datasets.ImageFolder(train_path, transform=trans)
    test_data = torchvision.datasets.ImageFolder(test_path, transform=trans)
    return (data.DataLoader(train_data, batch_size,
                            shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(test_data, batch_size,
                            shuffle=False,num_workers=get_dataloader_workers()))




