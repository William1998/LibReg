#!/usr/bin/python3
import cv2 as cv
import numpy
import socket
import struct

def AcceptImage(HOST = '192.168.43.79', PORT = 10000):
    buffersize = 65535
    
    #Create server object
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind( (HOST, PORT) )
    
    print("Now waitting for the frame")
    
    while True:
        data, address = server.recvfrom(buffersize) # receive image
        data = numpy.array(bytearray(data))
        imagedecode = cv.imdecode(data, 1)
        print("Received one frame")
        cv.imshow('frames', imagedecode)
        
        # Waitting for frame process function
        
        if cv.waitKey(1) == 27:
            break
        
    server.close()
    cv.destroyAllWindows()
if __name__ == '__main__':
    AcceptImage()