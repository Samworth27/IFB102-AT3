# https://stackoverflow.com/questions/58141153/how-do-i-isolate-handwritten-text-from-an-image-using-opencv-and-python


import numpy as np
import cv2 as cv


# https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size


def padding(array, xx, yy, value=0.5):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    if xx < w or yy < h:
        return array

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=value)

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


frame_size = (700, 700)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    ret, digit, annotated = detect_digit(frame)

    if ret is False:
        digit = np.zeros((28,28)).astype(np.uint8)
    
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
    
    
    cv.imshow('frame', np.concatenate((padding(annotated, *frame_size), padding(
        digit, *frame_size, 1)), axis=1))
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
