# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rec_UI.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(600, 300)
        Form.setStyleSheet(u"border-color: rgb(0, 0, 0);")
        self.lineEdit = QLineEdit(Form)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(120, 60, 211, 31))
        font = QFont()
        font.setPointSize(14)
        self.lineEdit.setFont(font)
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 60, 81, 31))
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(30, 110, 81, 41))
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.lineEdit_2 = QLineEdit(Form)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setGeometry(QRect(120, 110, 91, 31))
        self.lineEdit_2.setFont(font)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(120, 180, 91, 51))
        self.pushButton.setFont(font)
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(10, 180, 91, 51))
        self.pushButton_2.setFont(font)
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(360, 30, 211, 221))
        self.groupBox.setStyleSheet(u"")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 30, 171, 171))
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(230, 180, 91, 51))
        self.pushButton_3.setFont(font)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u56fe\u7247\u8bc6\u522b", None))
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("Form", u"\u8bf7\u8f93\u5165\u56fe\u7247\u8def\u5f84", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u56fe\u7247\u8def\u5f84", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u8bc6\u522b\u7ed3\u679c", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"\u8bc6\u522b", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"\u663e\u793a\u56fe\u7247", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"\u56fe\u7247\u663e\u793a", None))
        self.label_3.setText("")
        self.pushButton_3.setText(QCoreApplication.translate("Form", u"\u6e05\u7a7a", None))
    # retranslateUi

