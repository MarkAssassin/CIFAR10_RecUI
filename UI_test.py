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
import scipy.io

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QTableWidgetItem, QFileDialog
from PySide6.QtGui import QFont, QPixmap
from UI.rec_UI import Ui_Form

from model_test import rec_single_img


# pyside6-uic rec_UI.ui -o rec_UI.py

class recWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.bind() #绑定函数

    def bind(self):
        self.pushButton.clicked.connect(self.recognize)
        self.pushButton_2.clicked.connect(self.draw)
        self.pushButton_3.clicked.connect(self.clear)

    def recognize(self):
        img_path=self.lineEdit.text()
        pred=rec_single_img(img_path)
        self.lineEdit_2.setText(pred)


    def draw(self):
        img_path = self.lineEdit.text()
        self.label_3.setScaledContents(True)  # 设置缩放,即图片可自适应Label的大小
        self.label_3.setPixmap(QPixmap(img_path))

    def clear(self):
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.label_3.clear()


if __name__ == '__main__':
    app=QApplication([])
    window=recWindow()
    window.show()
    app.exec()


