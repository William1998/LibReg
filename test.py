from ObjDectector import ObjDector
<<<<<<< HEAD
from centroidtracker import CentroidTracker
from object_tracking import track_object
import cv2 as cv
import numpy as np

detectedObjects = []
ct = CentroidTracker()


<<<<<<< HEAD
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
=======
=======
import cv2 as cv
import numpy as np



>>>>>>> 1b5a30006579c93b409f27a7404112ede5b7c0b2
def test(image, tmp):
    
    items, cata, confidences, boxes = tmp.detect(image)
    tmp.drawBox(items, cata, confidences, boxes,image)    
<<<<<<< HEAD
>>>>>>> f3eeb19ab100a49a98512561fb90937182b89401
=======
>>>>>>> 1b5a30006579c93b409f27a7404112ede5b7c0b2



