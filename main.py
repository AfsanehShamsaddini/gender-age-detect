from PyQt5.uic.properties import QtGui
from cv2 import data
from PyQt5.QtGui import QPixmap, QImage
from  PyQt5.QtWidgets import  QMainWindow, QApplication,QLabel,QPushButton
from PyQt5 import uic
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi('detect.ui',self)
        self.label = self.findChild(QLabel,"label")
        self.image_label = self.findChild(QLabel, "detect")
        self.save_btn = self.findChild(QPushButton, "save")
        self.image_btn = self.findChild(QPushButton, "image")
        self.FACE_PROTO = "weights/opencv_face_detector.pbtxt"
        self.FACE_MODEL = "weights/opencv_face_detector_uint8.pb"

        self.AGE_PROTO = "weights/age_deploy.prototxt"
        self.AGE_MODEL = "weights/age_net.caffemodel"

        self.GENDER_PROTO = "weights/gender_deploy.prototxt"
        self.GENDER_MODEL = "weights/gender_net.caffemodel"

        self.FACE_NET = cv2.dnn.readNet(self.FACE_MODEL, self.FACE_PROTO)
        self.AGE_NET = cv2.dnn.readNet(self.AGE_MODEL, self.AGE_PROTO)
        self.GENDER_NET = cv2.dnn.readNet(self.GENDER_MODEL, self.GENDER_PROTO)
        self.MODEL_MEAN_VALUE = (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
        self.GENDER_LIST = ["Male", "Female"]

        self.box_padding = 20
        self.final=""
        self.show()

        self.image_btn.clicked.connect(self.image_detect)
        self.save_btn.clicked.connect(self.save_detect)
    def get_face_box(self,net, frame, conf_threshold = 0.7):
        frame_copy = frame.copy()
        frame_height = frame_copy.shape[0]
        frame_width = frame_copy.shape[1]

        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detection = net.forward()
        boxes = []

        for i in range(detection.shape[2]):
            confidence = detection[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detection[0, 0, i, 3] * frame_width)
                y1 = int(detection[0, 0, i, 4] * frame_height)
                x2 = int(detection[0, 0, i, 5] * frame_width)
                y2 = int(detection[0, 0, i, 6] * frame_height)
                boxes.append([x1,y1, x2, y2])
                cv2.rectangle(frame_copy, (x1,y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

        return frame_copy, boxes

    def image_detect(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Image files (*.jpg *.gif)")

        imagepath = fname[0]
        print(imagepath)
        img = cv2.imread(imagepath)
        resized_image = cv2.resize(img, (640, 480))
        frame = resized_image.copy()
        frame_face, boxes = self.get_face_box(self.FACE_NET, frame)
        for box in boxes:
            face = frame[max(0, box[1] - self.box_padding) : min(box[3] + self.box_padding, frame.shape[0] - 1),\
                   max(0, box[0] - self.box_padding): min(box[2] + self.box_padding, frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUE, swapRB= False)
            self.GENDER_NET.setInput(blob)
            gender_predictions = self.GENDER_NET.forward()
            gender = self.GENDER_LIST[gender_predictions[0].argmax()]
            print("Gender: {}, conf: {:.3f}".format(gender, gender_predictions[0].max()))

            self.AGE_NET.setInput(blob)
            age_predictions = self.AGE_NET.forward()
            age = self.AGE_LIST[age_predictions[0].argmax()]
            print("Age: {}, conf: {:.3f}".format(age, age_predictions[0].max()))

            label = "{}, {}".format(gender, age)
            cv2.putText(frame_face,label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite("Output.jpg", frame_face)
        self.final = frame_face
        self.pixmap_size = QSize(self.image_label.width(), self.image_label.height())
        self.pixmap = QPixmap("Output.jpg")
        self.pixmap = self.pixmap.scaled(self.pixmap_size)
        self.image_label.setPixmap(self.pixmap)

    def save_detect(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        # if file path is blank return back
        if filePath == "":
            return

        # saving canvas at desired path
        cv2.imwrite(filePath,self.final)


if __name__ == '__main__':
    app=QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()

