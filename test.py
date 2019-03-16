from ObjDectector import ObjDector
from centroidtracker import CentroidTracker
from object_tracking import track_object
import cv2 as cv
import numpy as np

detectedObjects = []
ct = CentroidTracker()


# while True:
x = ObjDector.ObjDector()

cap = cv.VideoCapture("./test/test1.jpeg")
retval, image = cap.read()
items, confidences, boxes = x.detect(image)
print(boxes[0])

detectedObjects = track_object(ct, items, [image[0], image[1]])
# print(detectedObjects)

    # x.drawBox(items, confidences, boxes, image)
    # cv.imwrite("./test/testresult.jpg", image.astype(np.uint8))



