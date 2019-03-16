import os
import cv2 as cv
class FaceRecgnizer():
    def __init__(self):
        self.modelPath = "faceModel/faceModel.XML"
        self.model = cv.face.FisherFaceRecognizer_create()
        if os.path.exists(self.modelPath):
            print("success")
        else:
            print("No model")

x = FaceRecgnizer()