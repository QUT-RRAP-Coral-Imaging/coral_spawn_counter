#! /usr/bin/env python3

"""
circle detector using Hough transforms
"""

import cv2 as cv
import os

# import matplotlib.pyplot as plt
import numpy as np


class CircleDetector:


# with the arguments:
# gray: Input image (grayscale).
# circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
# HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
# dp = 1: The inverse ratio of resolution.
# min_dist = gray.rows/16: Minimum distance between detected centers.
# param_1 = 200: Upper threshold for the internal Canny edge detector.
# param_2 = 100*: Threshold for center detection.
# min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
# max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.

# current settings work reasonably well for white background/red microspheres
    BLUR_DEFAULT = 5
    METHOD_DEFAULT = cv.HOUGH_GRADIENT_ALT
    DP_DEFAULT = 0.7
    MINDIST_DEFAULT = 40
    PARAM1_DEFAULT = 100
    PARAM2_DEFAULT = 20
    MAXRADIUS_DEFAULT = 70
    MINRADIUS_DEFAULT = 15

    def __init__(self,
                 blur = BLUR_DEFAULT,
                 method = METHOD_DEFAULT,
                 dp = DP_DEFAULT,
                 minDist = MINDIST_DEFAULT,
                 param1 = PARAM1_DEFAULT,
                 param2 = PARAM2_DEFAULT,
                 maxRadius = MAXRADIUS_DEFAULT,
                 minRadius = MINRADIUS_DEFAULT):

        self.blur = blur
        self.method = method
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.maxRadius = maxRadius
        self.minRadius = minRadius

        self.circles = []
        self.count = []


    def find_circles(self, img, det_param=None):

        method = self.method
        if det_param is None:
            blur = self.blur
            dp = self.dp
            minDist = self.minDist
            param1 = self.param1
            param2 = self.param2
            maxRadius = self.maxRadius
            minRadius = self.minRadius
        else:
            blur = det_param["blur"]
            dp = det_param['dp']
            minDist = det_param['minDist']
            param1 = det_param['param1']
            param2 = det_param['param2']
            maxRadius = det_param['maxRadius']
            minRadius = det_param['minRadius']

        # TODO rename parameters for more readability
        # convert image to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # blur image (Hough transforms work better on smooth images)
        # img = cv.medianBlur(img, blur)
        # hist equ
        # img = self.histogram_equalization(img)
        img = cv.medianBlur(img, blur)



        # find circles using Hough Transform
        circles = cv.HoughCircles(image = img,
                                method=method,
                                dp=dp,
                                minDist=minDist,
                                param1=param1,
                                param2=param2,
                                maxRadius=maxRadius,
                                minRadius=minRadius)      

        # import code
        # code.interact(local=dict(globals(), **locals())) 
        # remove duplicate circles, since min dist doesn't seem to be working
        # for similarly-sized circles:
        
        # for i, c in enumerate(circles[0,:]):
        # xc = circles[0][:,0].tolist()
        # yc = circles[0][:,1].tolist()
        # uxc = set(xc) # unique x's
        # uyc = set(yc) # unique y's
        if circles is not None:
            centres = circles[0][:,0:2].tolist()
            # nonoverlapping centres
            # nonunique_centre_indices = [c for c, i in enumerate(centres) if centres.count(i) > 1]
            # nonunique_centre = [c for i, c in enumerate(centres) if centres.count(c) > 1]
            
            unique_centres = []
            idx_unique = []
            for i, c in enumerate(centres):
                if c not in unique_centres: # TODO ideally, replace this with a small range
                    unique_centres.append(c)
                    idx_unique.append(i)
            circles_unique = circles[0][idx_unique,:]
            circles_unique = np.expand_dims(circles_unique, axis=0)

            return circles_unique
        else:
            return None


    def histogram_equalization(self, img):
        """ equalize histogram to increase global contrast of images, distribute intensities better"""
        return cv.equalizeHist(img)

    def draw_circles(self, img, circles, outer_circle_color=(0, 255, 0), thickness=8):
        """ draw circles onto image"""
        for circ, i in enumerate(circles[0,:], start=1):
            cv.circle(img, 
                    (int(i[0]), int(i[1])), 
                    radius=int(i[2]), 
                    color=outer_circle_color, 
                    thickness=thickness)
        return img


    def count_spawn(self, img, det_param):
        # proxy for ML detector, which will come later in the project
        # counts the spawn
        # saves a figure of the detections

        circles = self.find_circles(img, det_param)

        if circles is not None:
            _, count, _ = circles.shape
        else:
            count = 0
        self.circles = circles
        self.count = count
        return count, circles


    def save_detections(self, img, img_name, save_dir, resize):

        # resize to scale the image down - save memory/space

        if self.circles is not None:
            img_c = self.draw_circles(img, self.circles)
        else:
            img_c = img
            
        img_name_circle = img_name[:-4] + '_circ.png'
        # expect RGB, but since CV, need to save as BGR
        img_c = cv.cvtColor(img_c, cv.COLOR_RGB2BGR)

        img_s = cv.resize(img_c, (0, 0), fx=resize, fy=resize)

        cv.imwrite(os.path.join(save_dir, img_name_circle), img_s)