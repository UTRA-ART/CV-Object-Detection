import os
import time
import math
import copy

import cv2
import numpy as np

class State:
    ''' Maintains a 'memory' for the cascade classifier
    Allows for criteria between frames on whether or not to accept classifications '''
    def __init__(self):
        self.poi = np.empty([1,5])
        self.vor = None
        self.min_weight = 0.3
        self.max_distance = 200
        self.decay = 0.8

        fileDir = os.path.dirname(os.path.realpath('__file__')) # __file__ = main.py 
        self.ss_cascade = cv2.CascadeClassifier(os.path.join(fileDir, '.\models\stopsigns.xml'))

    def draw(self, img):
        ret = copy.copy(img)
        if len(self.poi) > 1:
            index = np.argmax(self.poi[:,4])
            if index > 0:
                [x, y, w, h, s] = self.poi[index]
                color = (255, 0, 0) if len(ret.shape) == 3 else 255
                cv2.rectangle(ret, (int(x), int(y)), (int(x + w), int(y + h)), color, math.ceil(s))
        return ret 

    def update(self, data):
        '''
        Alogirthm:
        1) check closest point of same type incoming
        2) increase both points by 1 if so 
        3) reduce weight and filter of all points 
        '''
        #Find near points 
        for d in data:
            for (x, y, w, h) in d:    
                if len(self.poi) > 0:
                    closest_index = np.argmin(np.sum((self.poi[:,0:2] - np.array([x,y]))**2, axis=1))

                    if np.sum((self.poi[closest_index,0:4] - np.array([x,y,h,w]))**2) < self.max_distance:
                        self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1 + self.poi[closest_index,4]])])
                        self.poi[closest_index,4] *= 0.5
                    else:
                        self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1])])
                else:
                    self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1])])

        i = 1
        while i < len(self.poi):
            self.poi[i,4] *= self.decay
            if self.poi[i,4] < self.min_weight:
                self.poi = np.delete(self.poi, i, axis=0)
            else:
                i += 1

    def run_cascades(self, img):  
        ss = self.ss_cascade.detectMultiScale(img, 2, 2)
        self.update([ss])
        
        return ss, [], [], []