from ObjDectector import ObjDector
import cv2 as cv
import numpy as np

x = ObjDector.ObjDector()

cap = cv.VideoCapture("./test/test1.jpg")
retval, image = cap.read()
items, confidences, boxes = x.detect(image)

x.drawBox(items, confidences, boxes,image)
cv.imwrite("./test/testresult.jpg", image.astype(np.uint8))



