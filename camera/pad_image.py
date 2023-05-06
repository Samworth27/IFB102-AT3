# https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size

import numpy as np

def pad_image(array, xx, yy, value=0.5):
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