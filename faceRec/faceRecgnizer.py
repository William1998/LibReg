import os
import cv2
class FaceRecgnizer():
    def __init__(self):
        self.modelPath = "faceModel/faceModel.XML"
        self.model = cv.face.FisherFaceRecognizer_create()
        if os.path.exists(self.modelPath):
            print()

x = FaceRecgnizer()