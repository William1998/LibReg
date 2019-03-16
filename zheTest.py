from ObjDectector import ObjDector
import cv2 as cv
import numpy as np
from faceRec import faceRecgnizer

# x = ObjDector.ObjDector("./pre-trained-model/yolov3.weights","./pre-trained-model/yolov3.cfg",288)
#
# cap = cv.VideoCapture("./test/test1.jpg")
# retval, image = cap.read()
# items,cata, confidences, boxes = x.detect(image)
#
# x.drawBox(detectedObjects,image)
# cv.imwrite("./test/testresult.jpg", image.astype(np.uint8))

x = faceRecgnizer.FaceRecgnizer()
face, confidence = x.predict("./test/WIN_20190317_01_23_33_Pro.jpg")
print(face,confidence)
