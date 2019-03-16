#!/usr/bin/python3
import cv2 as cv
import numpy
import socket
import struct
import test
from ObjDectector import ObjDector

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
    while True:
        data, address = server.recvfrom(buffersize) # receive image
        data = numpy.array(bytearray(data))
        imagedecode = cv.imdecode(data, 1)
        print("Received one frame")
        
        # Waitting for frame process function
        test.test(imagedecode, x)
        cv.imshow('frames', imagedecode)
        if cv.waitKey(1) == 27:
            break
        
    server.close()
    cv.destroyAllWindows()

AcceptImage()