import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import TensorDataset  # TensorDataset可以用来对tensor数据进行打包,该类中的 tensor 第一维度必须相等(即每一个图片对应一个标签)
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import scipy.io
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from PIL import Image

from get_model import LeNet

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

# class LeNet(nn.Module):                     # 继承于nn.Module这个父类
#     def __init__(self):                     # 初始化网络结构
#         super(LeNet, self).__init__()       # 多继承需用到super函数
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):			 # 正向传播过程
#         x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
#         x = self.pool1(x)            # output(16, 14, 14)
#         x = F.relu(self.conv2(x))    # output(32, 10, 10)
#         x = self.pool2(x)            # output(32, 5, 5)
#         x = x.view(-1, 32*5*5)       # output(32*5*5)
#         x = F.relu(self.fc1(x))      # output(120)
#         x = F.relu(self.fc2(x))      # output(84)
#         x = self.fc3(x)              # output(10)
#         return x

softmax=nn.Softmax() #在计算准确率时还是要加上Softmax的

model=LeNet()
model_weight_pth='./Model_pth/CIFAR10_15.pth'
model.load_state_dict(torch.load(model_weight_pth)) #使用model.state_dict方式保存，则需要这样下载网络
model.to(device)
# print(model)

def rec_single_img(data_path):
    data=Image.open(data_path).resize((200,200))
    data = data.convert('RGB')
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),  # 这里Resize是将图片的尺寸转为32*32，因为之前训练的网络对图片的要求就是32*32
         torchvision.transforms.ToTensor()])
    data=transform(data)
    data = torch.reshape(data, (1, 3, 32, 32))  # 需要的是4维数据，但图片是3维，用这行代码进行改写
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        output = model(data)
        output = softmax(output)
        # _, predicted = torch.max(output.data, 1)
        label= output.data.max(1, keepdims=True)[1]
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        predicted=classes[label]


    return predicted

# predicted=rec_single_img('image1.jpg')
# print(predicted)