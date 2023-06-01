# https://stackoverflow.com/questions/58141153/how-do-i-isolate-handwritten-text-from-an-image-using-opencv-and-python

import numpy as np
import cv2 as cv
from imutils.video import VideoStream

from .detect_digit import detect_digit
from .pad_image import pad_image

class Camera:
    def __init__(self):
        # self.cap = cv.VideoCapture(0)
        self.cap = VideoStream(usePiCamera=False).start()
        # if not self.cap.isOpened():
        #     print("Cannot open camera")
        #     exit()
        self.frame_size = (700,700)
        
    def capture_continuous(self):
        while True:
            # Capture frame-by-frame
            captured, detected, digit, annotated = self.capture()
            if not captured:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if not detected:
                digit = np.zeros((28,28)).astype(np.uint8)           
            
            self.display_capture(digit,annotated)
            
            if cv.waitKey(1) == ord('q'):
                break
            
        # When everything done, release the capture
        self.exit()
        
    def capture(self):
        # ret, frame = self.cap.read()
        frame = self.cap.read()
        ret = True
        if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return False, False, None, None
        ret, digit, annotated = detect_digit(frame)
        if ret is False:
                digit = np.zeros((28,28)).astype(np.uint8)
        return True, ret, digit, annotated
    
    def display_capture(self,digit,annotated):
        cv.imshow('frame', np.concatenate((pad_image(annotated, *self.frame_size), pad_image(
                digit, *self.frame_size, 1)), axis=1))
            
        
    def exit(self):
        self.cap.stop()
        cv.destroyAllWindows()