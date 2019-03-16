<<<<<<< HEAD
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
face, confidence = x.predict("./initialFaces/s3/WIN_20190317_00_02_50_Pro.jpg")




=======
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
<<<<<<< HEAD
x.predict("./initialFaces/s3/WIN_20190317_00_02_50_Pro.jpg")
=======
face, confidence = x.predict("./initialFaces/s3/WIN_20190317_00_02_50_Pro.jpg")

>>>>>>> 1b5a30006579c93b409f27a7404112ede5b7c0b2



>>>>>>> a3eee0e3bd6554c849ec8f0b7985c9f2337baac2
