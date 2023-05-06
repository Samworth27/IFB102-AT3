try: 
    import RPi.RPIO
    rp = True
except:
    rp = False
    
print(f"Running on a pi: {rp}")

import cv2 as cv
from camera.camera import Camera
from NN.network import Network
from seven_seg.driver import SevenSegment

sevseg = SevenSegment(7,11,13,15)

network = Network()
network.load_network('net1')
cam = Camera()
# cam.capture_continuous()
while True:
    captured,detected,digit,annotated = cam.capture()
    cam.display_capture(digit,annotated)
    predicted = network.predict([digit.reshape((1,28*28))])[0]
    sevseg.display(predicted)
    if cv.waitKey(1) == ord('q'):
        break
cam.exit()