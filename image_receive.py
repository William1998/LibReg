#!/usr/bin/python3
import cv2 as cv
import numpy
import socket
import struct
from ObjDectector import ObjDector
from object_tracking import track_object
from collections import OrderedDict
from centroidtracker import CentroidTracker

def SendFrame(host, port, image):
    server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server.connect((host, port))
    result, imgencode=cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,50])
    server.sendall(imgencode)

def AcceptImage(HOST = '192.168.43.79', PORT = 10000):
    buffersize = 65535
    
    #Create server object
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    server.bind( (HOST, PORT) )
    
    print("Now waitting for the frame")
    x = ObjDector.ObjDector('/home/wei/repos/LibReg/pre-trained-model/yolov3.weights', "/home/wei/repos/LibReg/pre-trained-model/yolov3.cfg", 288)
    detectedObjects = OrderedDict()
    ct = CentroidTracker()
    while True:
        data, address = server.recvfrom(buffersize) # receive image
        data = numpy.array(bytearray(data))
        imagedecode = cv.imdecode(data, 1)
        print("Received one frame")
        
        # Waitting for frame process function
        items, cata, confidence, boxed = x.detect(imagedecode)
        track_object(ct, detectedObjects, items, cata, [imagedecode.shape[0], imagedecode.shape[1]])
        x.drawBox(detectedObjects,imagedecode)
        cv.imshow('frames', imagedecode)
        if cv.waitKey(1) == 27:
            break
        
    server.close()
    cv.destroyAllWindows()

AcceptImage()