#! /usr/bin/env python3

"""
code pulled from detect_circle_annotations.py to be a detector
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
import torch
import time
from scipy.spatial import KDTree

from coral_spawn_counter.Detector import Detector
# import machinevisiontoolbox as mvt
# from machinevisiontoolbox.Image import Image as MvtImage
# from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.SubSurface_Detector import SubSurface_Detector

class RedCircle_Detector(Detector):
   
    DEFAULT_META_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/' # where the metadata is
    DEFAULT_IMG_DIR = os.path.join(DEFAULT_META_DIR, 'microspheres')
    DEFAULT_SAVE_DIR = '/home/cslics04/images/redcircles'

    DEFAULT_MAX_DECT = 10
    
    # TODO all the parameters for detection here as defaults
    
    def __init__(self,
                 meta_dir: str = DEFAULT_META_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_img: int = DEFAULT_MAX_DECT,
                 save_prelim_img: bool = True,
                 img_size: int = 1280):
        
        self.count = 0
        
        # self.input_image_size = 1280 # pixels
        self.conf_def = 0.5
        self.class_idx_def = 5 # eggs
        self.class_name_def = 5 # eggs
        

        Detector.__init__(self, 
                          meta_dir = meta_dir,
                          img_dir = img_dir,
                          save_dir = save_dir,
                          max_img = max_img,
                          save_prelim_img = save_prelim_img,
                          img_size=img_size)
        
        # circle detector parameters
        self.blur_circles = 5
        self.method = cv.HOUGH_GRADIENT
        self.accumulator_res = 0.25 # dp, This parameter is the inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details). Essentially, the larger the dp gets, the smaller the accumulator array gets.
        
        self.maxEdgeThresh = 150 # Gradient value used to handle edge detection in the Yuen et al. method.
        self.minEdgeThresh = 2 # The smaller the threshold is, the more circles will be detected (including false circles). The larger the threshold is, the more circles will potentially be returned.
        self.maxRadius = int(self.img_size/100) 
        self.minRadius = int(self.img_size/200)
        self.minDist = int(1.5*self.minRadius)

        # red threshold parameters
        self.blur_threshold: int = 3
        self.lab_thresh_lower: np.ndarray = np.array([50, 150, 150])
        self.lab_thresh_upper: np.ndarray = np.array([240, 255, 255])

        # circle morphology on red-thresholded image
        self.ksize_morph1 = 5 # based on 1280 image size currently, first spherical kernel size for initial eroding, opening/closing
        self.ksize_morph2 = 3 # based on 1280 image size currently, second spherical kernel size for finer eroding

        self.blob_area_max= 4000# 1000
        self.blob_area_min= 25
        self.blob_circ_min = 0.75
        self.blob_circ_max = 1.1
                                
    # TODO input config style for detector that sets all parameters

    # TODO getters and setters for circle detection parameters
    def get_blur_circles(self):
        return self.blur_circles
    
    def get_method(self):
        return self.method
    
    def get_accumulator_res(self):
        return self.accumulator_res
    
    def get_min_dist(self):
        return self.minDist
    
    def get_max_edge_thresh(self):
        return self.maxEdgeThresh
    
    def get_min_edge_thresh(self):
        return self.minEdgeThresh
    
    def get_max_radius(self):
        return self.maxRadius
    
    def get_min_radius(self):
        return self.minRadius

    def set_blur(self, blur_circles: float):
        self.blur_circles = blur_circles

    def set_accumulator_res(self, accumulator_res: float):
        self.accumulator_res = accumulator_res

    def set_min_dist(self, mindist: float):
        self.minDist = mindist

    def set_max_edge_thresh(self, thresh: float):
        self.maxEdgeThresh = thresh

    def set_min_edge_thresh(self, thresh: float):
        self.minEdgeThresh = thresh

    def set_max_radius(self, r:float):
        self.maxRadius = r
    
    def set_min_radius(self, r: float):
        self.minRadius = r

    #########################


    def threshold_red(self, 
                      img_rgb: np.ndarray): 
        """ threshold for red parts of the image """
        # convert to LAB colour space - for uniform colour perception in Euclidian space

        # assume image is input as a rgb image

        # first blur to reduce noise prior color conversion:
        img_rgb = cv.medianBlur(img_rgb, self.blur_threshold)

        # convert to LAB colour space, just need a-channel for red
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)

        # threshold lab image for red pixels
        img_lab_red = cv.inRange(img_lab, self.lab_thresh_lower, self.lab_thresh_upper)

        if self.save_prelim_img:
            cv.imwrite(os.path.join(self.save_dir, 'threshold_red.jpg'), img_lab_red)

        return img_lab_red


    def morph_to_circles(self, img: np.ndarray, iter=2):
        """ morphological operations on binary images to make more circle-like"""
        
        # should be on resized image
        # kernel = np.ones((15,15), np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(self.ksize_morph1,self.ksize_morph1))
        img = cv.erode(img, kernel, iterations=iter)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

        img = cv.dilate(img, kernel, iterations=iter)

        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(self.ksize_morph2,self.ksize_morph2))
        img = cv.erode(img, kernel2, iterations=iter)

        if self.save_prelim_img:
            cv.imwrite(os.path.join(self.save_dir, 'morph_circles.jpg'), img)

        return img
        

    def find_blobs(self, img):
        """ alternatively, find blobs approach (similar to sub-surface detector)"""
        SD = SubSurface_Detector(img_dir=self.img_dir,
                                 meta_dir=self.meta_dir,
                                 save_dir=self.save_dir,
                                 save_prelim_img=True,
                                 area_max=self.blob_area_max,
                                 area_min=self.blob_area_min,
                                 circ_min=self.blob_circ_min,
                                 circ_max=self.blob_circ_max)
        blobby, icont = SD.attempt_blobs(img)
        return blobby, icont


    def find_circles(self, img):
        """
        Find circles in a given image with specified parameters
        img - as numpy array
        blur - kernel size to apply median blur to image (smoother image = better circles)
        dp - inverse ratio  of the accumulator resolution, dp=1: the accumulator is same res as img, dp=2: accumulator is half as big width, height
        mindist - min dist between detected circle centres
        # NOTE names aren't exactly correct for the Canny's hysteresis threshold procedure
        maxEdgeThresh - higher threshold of 2 thresholds, passed to Canny edge detector
        minEdgeThresh - lower threshold of 2 thresholds passed to Canny edge detector
        minRadius - min radius of circle
        maxRadius - max radius of circle
        """               
        # equalize histogram
        img = cv.equalizeHist(img)

        # blur image (Hough transforms work better on smooth images)
        img = cv.medianBlur(img, self.blur_circles)
        # find circles using Hough Transform
        circles = cv.HoughCircles(image = img,
                                    method=self.method,
                                    dp=self.accumulator_res,
                                    minDist=self.minDist,
                                    param1=self.maxEdgeThresh,
                                    param2=self.minEdgeThresh, 
                                    maxRadius=self.maxRadius,
                                    minRadius=self.minRadius)
        
        # TODO filter circles based on overlap?
        return circles


    def draw_circles(self, img, circles, outer_circle_color=(255, 0, 0), thickness=2):
        """ draw circles onto image"""
        # img can be 1 channel or 3 channels
        # if img is 1-channel, we make it 3-channels so we can colour our cicles differently to the b/w image
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for i, circ in enumerate(circles[0,:], start=1):
            cv.circle(img, 
                    (int(circ[0]), int(circ[1])), 
                    radius=int(circ[2]), 
                    color=outer_circle_color, 
                    thickness=thickness)
        return img


    def match_circles_to_blobs(self, circles, blobs, icont):
        """ match circles to blobs/icont, if circles and blobs match, then we have a red circle"""

        # do correspondence based on nearest centroid
        # if they don't match to within X threshold, then rejected
        # final list of circles is output
        # print('match_circles_to_blobs')

        # circles
        circles_sq = np.squeeze(circles)
        n_circ, _ = circles_sq.shape

        if n_circ > 0:
            centroids_circles = circles_sq[:,0:2] # x,y centroids, last column is radius

            x_b, y_b = blobs.centroid
            x_bi = np.array([x for i, x in enumerate(x_b) if i in icont])
            y_bi = np.array([y for i, y in enumerate(y_b) if i in icont])
            centroids_blobs = np.vstack((x_bi, y_bi)).transpose()

            kdtree = KDTree(centroids_circles)

            # blobs
            # for each point in the second list, find the closest point in the first
            # TODO set distance upperbound
            matching_indices = kdtree.query(centroids_blobs)[1]

            corresponding_centroids = centroids_circles[matching_indices]
            corresponding_radii = circles_sq[matching_indices, 2]

            # TODO can also check similar radii if need to double double check
            
            matching_circles = np.vstack((corresponding_centroids.transpose(), corresponding_radii.transpose())).transpose()        

            filtered_circles = np.expand_dims(matching_circles, axis=0)  
        else:
            # no circles to speak of
            filtered_circles = circles      
        return filtered_circles


    def convert_circle_to_box(self, x,y,r, image_width, image_height):
        """ convert a circle to a box, maxing sure to stay within the image bounds"""
        xmin = max(0, x - r)
        xmax = min(image_width, x + r)
        ymin = max(0, y - r)
        ymax = min(image_height, y + r)
        return xmin, ymin, xmax, ymax


    def detect(self, image: np.ndarray):
        """
        return detections from a single prepared image, input as a numpy array
        attempts to find circles and then converts the circles into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """

        # convert colour image to grayscale
        # if len(image.shape) == 3:
        #     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        img_height, img_width,_ = image.shape    
        
        # resize image to small size/consistency (detection parameters vs image size) and speed!
        # img_h, img_w, n_channels = img.shape
        img_scale_factor: float = self.img_size / img_width
        # print(f'img_scale_factor = {img_scale_factor}')
        image_resized: np.ndarray = cv.resize(image, None, fx=img_scale_factor, fy=img_scale_factor)
        # if len(image.shape) == 2:
        img_h_resized, img_w_resized, _ = image_resized.shape
        # else:
        #     img_h_resized, img_w_resized, _ = image.shape
     
        # threshold image based on CieLab red threshold
        imgt = self.threshold_red(image_resized)
     
        # morphological operations to refine the thresholded red image
        imgm = self.morph_to_circles(imgt)

        # find circles in binary image
        circles = self.find_circles(imgm)
        if circles is not None:
            _ , n_circ, _ = circles.shape
        else:
            n_circ = 0
        print(f'Circles detected = {n_circ}')
        
        if self.save_prelim_img:
            if n_circ > 0:
                img_c = self.draw_circles(imgm, circles)
            else:
                img_c = image_resized
            # save image
            img_name = self.img_list[self.count]
            img_name_circle = os.path.basename(img_name[:-4]) + '_circ_prematch.jpg'
            img_c = cv.cvtColor(img_c, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(self.save_dir, img_name_circle), img_c)


        blobs, icont = self.find_blobs(imgm)

        # now, we need a combine/agree on blobs and circles
        # match centroids and compare?
        if n_circ > 0:
            filtered_circles = self.match_circles_to_blobs(circles, blobs, icont)
            # TODO filter the zero case
            _, n_circ, _ = filtered_circles.shape
        else:
            filtered_circles = circles

        # save image with circles drawn ontop
        if self.save_prelim_img:
            if n_circ > 0:
                img_c = self.draw_circles(image_resized, filtered_circles)
            else:
                img_c = image_resized
            # save image
            img_name = self.img_list[self.count]
            img_name_circle = os.path.basename(img_name[:-4]) + '_circ.jpg'
            img_c = cv.cvtColor(img_c, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(self.save_dir, img_name_circle), img_c)

        # get bounding box of the circles
        pred = []
        for i in range(n_circ):
            x = filtered_circles[0][i, 0]
            y = filtered_circles[0][i, 1]
            r = filtered_circles[0][i, 2]
            xmin, ymin, xmax, ymax = self.convert_circle_to_box(x,y,r,img_w_resized, img_h_resized)
            
            # size-up boxes to original image sizes
            x1 = xmin/img_w_resized
            y1 = ymin/img_h_resized
            x2 = xmax/img_w_resized
            y2 = ymax/img_h_resized
            # print(xmin, ymin, xmax, ymax)
            pred.append([x1, y1, x2, y2, self.conf_def, self.class_idx_def, self.class_name_def]) # class definitions hard-coded here
        
        # print('predictions as tensors')
        return torch.tensor(pred)


    def run(self):   
        imgsave_dir = os.path.join(self.save_dir, 'detection_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detection_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)
        
        start_time = time.time()
        for i, img_name in enumerate(self.img_list):
            if i > self.max_img:
                print('hit max detections')
                break
            
            print(f'{i}: img_name = {img_name}')  
            # read in image
            img = self.prep_img_name(img_name)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # detect circles
            predictions = self.detect(img)
            self.count += 1
            self.save_image_predictions(predictions, img, img_name, imgsave_dir)
            self.save_text_predictions(predictions, img_name, txtsavedir)
            
        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))
        
        print(f'time[s]/image = {duration / len(self.img_list)}')
        

def main():
    # meta_dir = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    # img_dir = '/home/cslics04/20231018_cslics_detector_images_sample/microspheres'

    meta_dir = '/home/cslics/Code/cslics_ws/coral_spawn_imager'
    img_dir = '/home/cslics/Data/20231018_cslics_detector_images_sample/microspheres'
    save_dir = '/home/cslics/Data/20231018_cslics_detector_images_sample/output'
    
    max_dect = 20 # number of images
    Coral_Detector = RedCircle_Detector(meta_dir=meta_dir, img_dir = img_dir, save_dir=save_dir, max_img=max_dect)
    Coral_Detector.run()
    
    
    import code
    code.interact(local=dict(globals(), **locals()))

    # import matplotlib.pyplot as plt
    # plt.imshow(img_lab_red)
    # plt.show()


if __name__ == "__main__":
    main()