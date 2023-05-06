import numpy as np
import cv2 as cv

def detect_digit(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    threshold = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 8)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.dilate(threshold, kernel, iterations=6)
    contours, hierarchy = cv.findContours(
        dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    for contour in contours:
        x, y, width, height = cv.boundingRect(contour)
        print(x,y,width,height)
        half_width = width//2
        half_height = height//2
        x_centre = x+half_width
        y_centre = y+half_height
        img_size = max(width,height)//2
        ROI = (gray[y_centre-img_size:y_centre+img_size, x_centre - img_size:x_centre+img_size]).astype(np.uint8)
        # If the ROI is not a valid shape return just the grayscale frame
        if min(ROI.shape) == 0:
            return False, None, gray      
        ret, ROI = cv.threshold(ROI,120,255,cv.THRESH_BINARY_INV)
        # Resize to (20,20) then to (28,28) with AA, in order to match the process for creating the MNIST data set
        digit = cv.resize(ROI,(20,20),interpolation=cv.INTER_AREA)
        digit = cv.resize(digit,(28,28),interpolation=cv.INTER_AREA)
        img_annotated = np.copy(gray)
        return True, digit, cv.rectangle(img_annotated, (x_centre - img_size, y_centre - img_size), (x_centre+img_size, y_centre+img_size), (255, 0, 0), 2)
    # A catch all
    return False, None, gray