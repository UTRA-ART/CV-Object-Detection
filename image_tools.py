  
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

def get_bbox(img):
    '''
    Returns bounding polygons for the all identified middles 
    in the image
    '''
    temp = gen_mask(img, False, True)

    # cv2.imshow("temp", temp)
    # cv2.waitKey(0)
    bound_poly, contours = thresh_callback(temp)

    return bound_poly, contours, temp

def gen_mask(img, bitwise_and=False, process=True):
    '''
    Masks input img based off HSV colour ranges provided 
    '''
    #Masks colour ranges provided
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask0 = cv2.inRange(hsv, LOWER_MASK1, UPPER_MASK1)
    mask1 = cv2.inRange(hsv, LOWER_MASK2, UPPER_MASK2)
    mask = mask0 + mask1

    cv2.bitwise_not(mask)

    if process:
        # Expands border to ensure when dilating shape is maintained 
        mask = cv2.copyMakeBorder(mask, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=0)

        kernel = np.ones((13,13), np.uint8) #0.05
        refined = cv2.erode(mask, kernel)

        kernel = np.ones((51,51), np.uint8) #0.05
        refined = cv2.dilate(refined, kernel)

        kernel = np.ones((51,51), np.uint8) #0.05
        refined = cv2.erode(refined, kernel)

        kernel = np.ones((13,13), np.uint8) #0.05
        refined = cv2.dilate(refined, kernel)

    else:
        refined = mask

    if bitwise_and:
        # print(img.shape, refined[150:-150,150:-150].shape)
        return cv2.bitwise_and(img, img, mask=refined[150:-150,150:-150])
    return refined[151:-151,151:-151]

def thresh_callback(mask):
    ''' Returns single convex hull of all contours of mask '''   

    # print(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 0):
        return 0, 0

    minRects = [None]*len(contours)
    for i, c in enumerate(contours):
        temp = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        minRects[i] = [temp, cv2.moments(temp)]

    filt = []
    for i in range(0, len(minRects)):
        flag = True
        if (minRects[i][1]['m00'] < MIN_AREA):
            flag = False
            
        filt += [flag]

    ret = [minRects[i] for i in range(0, len(minRects)) if filt[i]==True]

    # print("BOX", ret)
    return ret, contours