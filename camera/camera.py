# https://stackoverflow.com/questions/58141153/how-do-i-isolate-handwritten-text-from-an-image-using-opencv-and-python

import numpy as np
import cv2 as cv

from .detect_digit import detect_digit
from .pad_image import pad_image

class Camera:
    def __init__(self):
        self.cap = cv.VideoCapture("/dev/video2")
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
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
        ret, frame = self.cap.read()
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
        self.cap.release()
        cv.destroyAllWindows()


# old capture code
# digit_normalised = cv.normalize(
#     digit, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)

# gray_normalised = cv.normalize(
#     annotated, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)

# squared = square_image(digit_normalised)

#     max_size = min(min(gray.shape),max(normalised.shape))
#     print(max_size)
#     square = padding(normalised, max_size,max_size, mode='minimum')
# else:
#     square = normalised

# Display the resulting frame
# row_1 = np.concatenate((blur, threshold, dilate), axis=1)
# row_2 = np.concatenate(
#     (padding(ROI, *gray.shape), padding(normalised, *gray.shape), gray_annotated), axis=1)