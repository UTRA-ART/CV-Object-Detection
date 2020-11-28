  
import numpy as np
import cv2

MIN_AREA = 0
LOWER_MASK1 = np.array([0,50,50])
UPPER_MASK1 = np.array([10,255,255])
LOWER_MASK2 = np.array([170,50,50])
UPPER_MASK2 = np.array([180,255,255])

def crop(img):
    '''Crops image to be largest square possible'''
    iH, iW, _ = img.shape
    crop_size = min(iH, iW)
    ret = img[(iH - crop_size)//2:(iH - crop_size)//2 + crop_size,(iW - crop_size)//2:(iW - crop_size)//2+crop_size]
    return ret

def scale(img, width=250):
    iH, iW, _ = img.shape

    final_window_width = width

    dest_width = final_window_width
    dest_height = round((final_window_width / iW) * iH)

    res = cv2.resize(img, dsize=(dest_width, dest_height), interpolation=cv2.INTER_CUBIC)
    return res