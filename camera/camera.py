# https://stackoverflow.com/questions/58141153/how-do-i-isolate-handwritten-text-from-an-image-using-opencv-and-python

import numpy as np
import cv2 as cv

from detect_digit import detect_digit
from pad_image import pad_image

class Camera:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.frame_size = (700,700)
        
    def capture(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            ret, digit, annotated = detect_digit(frame)

            if ret is False:
                digit = np.zeros((28,28)).astype(np.uint8)           
            
            cv.imshow('frame', np.concatenate((pad_image(annotated, *self.frame_size), pad_image(
                digit, *self.frame_size, 1)), axis=1))
            
            if cv.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
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