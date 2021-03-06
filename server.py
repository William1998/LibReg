# -*- coding: utf-8 -*-
import cv2 as cv
import numpy
import socket
import struct

HOST='192.168.43.79'
PORT=10000

server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket对象
server.connect((HOST,PORT))
print('now starting to send frames...')
capture= cv.VideoCapture(0)
#capture.set(cv.CAP_PROP_BUFFERSIZE, 3)
try:
    while True:
        success,frame=capture.read()
        while not success and frame is None:
            success,frame=capture.read()  #获取视频帧
        result,imgencode=cv.imencode('.jpg',frame,[cv.IMWRITE_JPEG_QUALITY,50])  #编码
        #server.sendall(struct.pack('i',imgencode.shape[0])) #发送编码后的字节长度，这个值不是固定的
        server.sendall(imgencode) #发送视频帧数据
        print('have sent one frame')
except Exception as e:
    print(e)
    #server.sendall(struct.pack('c',2)) #发送关闭消息
    capture.release()
    server.close()
