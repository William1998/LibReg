#!/usr/bin/python3
import cv2 as cv
import numpy
import socket
import struct
from ObjDectector import ObjDector
from object_tracking import track_object
from collections import OrderedDict
from centroidtracker import CentroidTracker
from faceRec import faceRecgnizer

def SendFrame(host, port, image):
    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.connect((host, port))
    result, imgencode=cv.imencode('.jpg',image,[cv.IMWRITE_JPEG_QUALITY,50])
    server.sendall(image)

def AcceptImage(HOST = 'localhost', PORT = 10000):
    buffersize = 65535 // 3
    
    #Create server object
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    server.bind( (HOST, PORT) )
    nameCount = 3
    with open("count") as f:
        count = f.read()
        count = int(count)
        nameCount =count


    faceRec = faceRecgnizer.FaceRecgnizer()
    
    print("Now waitting for the frame")
    x = ObjDector.ObjDector('pre-trained-model/yolov3.weights', "pre-trained-model/yolov3.cfg", 416)

    #Global things
    detectedObjects = OrderedDict()
    strangerList = OrderedDict()
    faceList = OrderedDict()
    ct = CentroidTracker()

    while True:
        data, address = server.recvfrom(buffersize) # receive image
        data = numpy.array(bytearray(data))
        imagedecode = cv.imdecode(data, 1)
        
        # Waitting for frame process function
        items, cata, confidence, boxed = x.detect(imagedecode)
        track_object(ct, detectedObjects, items, cata, [imagedecode.shape[0], imagedecode.shape[1]],\
                     imagedecode,faceRec,x,strangerList,nameCount,faceList)
        x.drawBox(detectedObjects,imagedecode,faceList)
        #SendFrame('192.168.43.160', 8080, imagedecode)
        cv.imshow('frames', imagedecode)
        if cv.waitKey(1) == 27:
            break
        
    server.close()
    cv.destroyAllWindows()