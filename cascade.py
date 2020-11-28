import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
import math


class State:
    def __init__(self):
        self.poi = np.empty([1,5])
        self.vor = None
        self.min_weight = 0.3
        self.max_distance = 1000000000000000
        self.decay = 0.8

    def draw(self, img):
        if len(self.poi) > 1:
            index = np.argmax(self.poi[:,4])
            if index > 0:
                [x, y, w, h, s] = self.poi[index]
                color = (255, 0, 0) if len(img.shape) == 3 else 255
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, math.ceil(s))

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

curr_state = State()
ss_cascade = cv2.CascadeClassifier('stopsigns.xml')
triangle_cascade = cv2.CascadeClassifier('triangles.xml')
traffic_light_cascade = cv2.CascadeClassifier('traffic_lights.xml')

def identify_signs(img):
    global curr_state, ss_cascade, triangle_cascade, traffic_light_cascade
    cascade_data = []

    start = time.time()

    ss = ss_cascade.detectMultiScale(img, 1.3, 2)
    # tria = triangle_cascade.detectMultiScale(img, 1.3, 5)
    # traf = traffic_light_cascade.detectMultiScale(img, 1.3, 5)

    curr_state.update([ss])
    curr_state.draw(img)
