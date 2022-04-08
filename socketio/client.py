import cv2
import socketio #python-socketio by @miguelgrinberg
import base64
import time

sio = socketio.Client()
sio.connect('http://localhost:4333')

cam = cv2.VideoCapture(0)
i = 0
while (True):
    i += 1
    ret, frame = cam.read()                     # get frame from webcam
    res, frame = cv2.imencode('.jpg', frame)    # from image to binary buffer
    data = base64.b64encode(frame)              # convert to base64 format
#   print('data : {}'.format(data))
    
    sio.emit('data', data)                      # send to server

cam.release()
# count = 0
# while True:
#     time.sleep(3)
#     count += 1
#     sio.emit('hellopython', 'ffffff')
#     print('i : {}'.format(count))