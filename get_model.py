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

#搭建神经网络
#in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2(padding：是否会在图片旁边进行填充)
class LeNet(nn.Module):                     # 继承于nn.Module这个父类
    def __init__(self):                     # 初始化网络结构
        super(LeNet, self).__init__()       # 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):			 # 正向传播过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

if __name__ == '__main__':

    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    # mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    # std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    # transforms_fn=torchvision.transforms.Compose([
    #     torchvision.transforms.RandomCrop(size=32,padding=4), #扩充为40*40，然后随机裁剪成32*32
    #     torchvision.transforms.RandomHorizontalFlip(0.5),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean, std)
    # ])
    transforms_fn=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #准备数据集
    #训练数据集
    train_data=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    #测试数据集
    test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

    # print(type(train_data))

    #获取2个数据集的长度
    train_data_size=len(train_data)
    test_data_size=len(test_data)
    print("训练数据集的长度为{}".format(train_data_size))
    print("测试数据集的长度为{}".format(test_data_size))

    #利用dataloader来加载数据集
    train=DataLoader(train_data,batch_size=64, shuffle=True)
    test=DataLoader(test_data,batch_size=64)



    softmax=nn.Softmax() #在计算准确率时还是要加上Softmax的

    model=LeNet()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    learning_rate = 0.001

    # 注意，1.如果用SGD优化器，一般学习率大一点，如0.1、0.05；Adam优化器学习率可以小一点,如0.001、0.0001
    #      2.最好加上weight_decay和学习率衰减
    # optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate,momentum=0.5,weight_decay=0.01)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate) # weight_decay越大，会进行惩罚，降低变化的速率，一般优化器中的参数都为默认值
    # scheduler=lr_scheduler.ExponentialLR(optimizer,gamma=0.94) #学习率衰减(指数衰减)，gamma不要设置的太小

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    epochs = 15


    #开始训练+测试
    for epoch in range(epochs):


        print("-----第{}轮训练开始------".format(epoch + 1))
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0, 0, 0, 0
        # arr=np.zeros((1,3630))
        test_label=[]
        test_pred=[]

        # 训练步骤开始
        model.train()
        for batch_idx, data in enumerate(train):
            inputs, labels = data
            inputs, labels=inputs.to(device),labels.to(device)

            # labels = torch.tensor(labels, dtype=torch.long)  # 需要将label转换成long类型
            optimizer.zero_grad()
            outputs = model(inputs)  # 需要加.float()，否则会报错
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算每轮训练集的Loss
            train_loss += loss.item()

            # 计算每轮训练集的准确度
            outputs = softmax(outputs)  # 计算准确率时需要添加softmax
            _, predicted = torch.max(outputs.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
            train_cor += (predicted == labels).sum().item()  # 正确分类个数
            train_sum += labels.size(0)  # train_sum+=predicted.shape[0]
        # scheduler.step()
            # train_true.append(labels.cpu().numpy()) #想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要先将它转化为 CPU tensor
            # train_pred.append(predicted.cpu().numpy())

        # 测试步骤开始
        # model.eval()
        # with torch.no_grad(): #即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导。
        #     for batch_idx1, data in enumerate(test):
        #         inputs, labels = data
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         # labels = torch.tensor(labels, dtype=torch.long)
        #
        #         outputs = model(inputs)
        #         loss = loss_fn(outputs, labels)
        #
        #         # 计算每轮训练集的Loss
        #         test_loss += loss.item()
        #
        #         # 计算每轮训练集的准确度
        #         outputs = softmax(outputs)  # 计算准确率时需要添加softmax
        #         _, predicted = torch.max(outputs.data, 1)
        #         test_cor += (predicted == labels).sum().item()
        #         test_sum += labels.size(0)
        #         # print(predicted)
        #         # print(labels)
        #
        #         # test_label = test_label + list(labels.cpu().numpy())
        #         # test_pred = test_pred + list(predicted.cpu().numpy())
        #
        #
        print("Train loss:{}   Train accuracy:{}%".format(train_loss / batch_idx, 100 * train_cor / train_sum))
        train_loss_list.append(train_loss / batch_idx)
        train_acc_list.append(100 * train_cor / train_sum)
        # test_acc_list.append(100 * test_cor / test_sum)
        # test_loss_list.append(test_loss / batch_idx1)

    # 保存网络
    torch.save(model.state_dict(), "CIFAR10_{}.pth".format(epochs)) #model.state_dict保存模型参数、超参数以及优化器的状态信息
    # torch.save(model, "CIFAR10_{}.pth".format(epochs))


    #输出混淆矩阵
    # print(len(test_label))
    # print(len(test_pred))
    # # print(test_label)
    # test_true=list(test_label.numpy())
    # # print(test_true)
    # # print(test_pred)
    # # print(type(test_true))
    # # print(type(test_pred))
    #
    # test_true=np.array(test_true) #强制转换为numpy的array形式
    # test_pred=np.array(test_pred)
    #
    # print(test_true)
    # print(test_pred)
    # print(test_true.shape)
    # print(test_pred.shape)
    #
    # C=confusion_matrix(test_true, test_pred) #混淆矩阵
    # print(C)
    # plt.matshow(C, cmap=plt.cm.Reds)
    # plt.show()

    #绘制曲线
    # fig = plt.figure()
    # plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    # plt.title('TrainLoss', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.grid()
    # plt.savefig('TrainLoss_HRRP')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
    # plt.title('TestLoss', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.grid()
    # plt.savefig('TestLoss_HRRP')
    # plt.show()

    fig = plt.figure()
    plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
    plt.title('TrainAccuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.grid()
    # plt.savefig('TrainAccuracy_HRRP')
    plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
    # plt.title('TestAccuracy', fontsize=14)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Accuracy(%)', fontsize=14)
    # plt.grid()
    # # plt.savefig('TestAccuracy_HRRP')
    # plt.show()