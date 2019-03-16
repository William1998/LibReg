from ObjDectector import ObjDector
import cv2 as cv
import numpy as np



def test(image, tmp):
    
    items, cata, confidences, boxes = tmp.detect(image)
    tmp.drawBox(items, cata, confidences, boxes,image)    



