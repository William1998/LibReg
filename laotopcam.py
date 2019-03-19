import numpy as np
import cv2
import socket

cap = cv2.VideoCapture(0)
HOST='localhost'
PORT=10000

server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket对象
server.connect((HOST,PORT))
print('now starting to send frames...')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.rotate(frame,1,frame)
    # Display the resulting frame
    result, imgencode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 编码
    # server.sendall(struct.pack('i',imgencode.shape[0])) #发送编码后的字节长度，这个值不是固定的
    server.sendall(imgencode)  # 发送视频帧数据
    print('have sent one frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
server.close()