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
# x.drawBox(items,cata, confidences, boxes,image)
# cv.imwrite("./test/testresult.jpg", image.astype(np.uint8))

x = faceRecgnizer.FaceRecgnizer()
x.predict("./initialFaces/s3/WIN_20190317_00_02_50_Pro.jpg")
face, confidence = x.predict("./initialFaces/s3/WIN_20190317_00_02_50_Pro.jpg")



