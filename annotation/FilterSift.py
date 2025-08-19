#!/usr/bin/env python3

# inspired from test_sift_features.py
# sift packaged as an object class with parameters, etc
# to be incorporated into annotation pipeline

import os
import glob
import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
from FilterCommon import FilterCommon

CONTRAST_THRESHOLD=0.03
EDGE_THRESHOLD=200
SIGMA=1.2
MIN_SIZE=10
MAX_SIZE=200
DILATE=100

DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21
DENOISE_STRENGTH = 5

class FilterSift(FilterCommon):
    
    def __init__(self,
                 contrast_threshold: float = CONTRAST_THRESHOLD,
                 edge_threshold: float = EDGE_THRESHOLD,
                 sigma: float = SIGMA,
                 min_size: float = MIN_SIZE,
                 max_size: float = MAX_SIZE,
                 dilate: int = DILATE,
                 template_window_size: int = DENOISE_TEMPLATE_WINDOW_SIZE,
                 search_window_size: int = DENOISE_SEARCH_WINDOW_SIZE,
                 denoise_strength: float = DENOISE_STRENGTH,
                 config: dict = None):
        
        if config:
            FilterCommon.__init__(self)
        
            # denoise parameters
            self.template_window_size = config['denoise_template_window_size']
            self.search_window_size = config['denoise_search_window_size']
            self.denoise_strength = config['denoise_strength']
            
            # sift feature parameters
            self.contrast_threshold = config['contrast_threshold']
            self.edge_threshold = config['edge_threshold']
            self.sigma = config['sigma']
            
            # filter sift features
            self.min_size = config['min_size']
            self.max_size = config['max_size']
            self.dilate = config['dilate']
        else:
            FilterCommon.__init__(self)
            
            # denoise parameters
            self.template_window_size = template_window_size
            self.search_window_size = search_window_size
            self.denoise_strength = denoise_strength
            
            # sift feature parameters
            self.contrast_threshold = contrast_threshold
            self.edge_threshold = edge_threshold
            self.sigma = sigma
            
            # filter sift features
            self.min_size = min_size
            self.max_size = max_size
            
            # when converting sift features to binary mask of putative coral matches
            # how much to dilate around the given features
            # NOTE could consider doing a variable dilation wrt size, but some features have not matched according to size/visually
            # instead, we choose a constant value appropriate for the size of the corals in the image (max radius) in units of pixels
            self.dilate = dilate
        
        # contrastThreshold	The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
        # edgeThreshold	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        # The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        self.sift = cv.SIFT_create(contrastThreshold=self.contrast_threshold, 
                                   edgeThreshold=self.edge_threshold, 
                                   sigma=self.sigma)
        
    
    def get_keypoints(self, image, mask=None):
        # assume image coming in is bgr
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        image_denoise = cv.fastNlMeansDenoising(image_gray, 
                                                templateWindowSize=self.template_window_size,
                                                searchWindowSize=self.search_window_size,
                                                h=self.denoise_strength)
        # NOTE: mask can be a binary area of where to look for keypoints, and is optional in the image
        # TODO checks for valid image, numpy array
        keypoints = self.sift.detect(image_denoise, mask)
        return keypoints
        
    
    def filter_size(self, keypoints):
        # print(f' length of kp before size filter: {len(keypoints)}')
        kp = []
        for k in keypoints:
            if k.size > self.min_size and k.size < self.max_size:
                kp.append(k)
        # print(f' length of kp after size filter: {len(keypoints)}')
        return kp
        
    
    def filter_response(self, keypoints):
        # take the keypoint with the highest response. We don't care about angle
        best_kp = {}
        for k in keypoints:
            if k.pt in best_kp: # xy tuple
                # check the existing keypoint is in the dictionary
                existing_kp = best_kp[k.pt]
                # if new kp is the same, but has a greater response, replace it
                if k.response > existing_kp.response:
                    best_kp[k.pt] = k
            else:
                best_kp[k.pt] = k
        return list(best_kp.values())
    
    
    def print_keypoints(self, kp):
        for i, k in enumerate(kp):

            print(f'{i}: pt = {k.pt}, size = {k.size}, resp = {k.response}, angle = {k.angle}')
            
    
    def draw_keypoints(self, image_bgr, kp):
        # image should be in bgr format
        
        color = (0, 0, 255) # red in bgr format
        img_ftr = cv.drawKeypoints(image_bgr, 
                                   kp,
                                   0, 
                                   color,
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_ftr
        
        
    def get_best_sift_features(self, image):
        kp = self.get_keypoints(image)
        # temp hack - trying to figure out why it's missing some very obvious keypoints
        # kp = self.filter_size(kp)
        kp = self.filter_response(kp)
        return kp
    
    
    def create_sift_mask(self, image, kp):
        # in areas around SIFT features in kp, create binary mask that says areas around here should be considered as putative matches for corals
        # we do this, so that we can just AND binary masks from other steps in the pipeline that are mask-oriented 
        
        # create mask
        # for each kp, draw on mask with specified dilation radius
        # return mask
        
        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        for k in kp:
            center = (int(k.pt[0]), int(k.pt[1]))
            radius = int(self.dilate)  # Ensure radius is an integer
            color = 255 # binary mask, white
            thickness = -1 # filled circle

            cv.circle(mask, center, radius, color, thickness)
        
        return mask
        
    
if __name__ == "__main__":
    print('SiftFeatures.py')
        
    img_pattern = '*.jpg'
    # img_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/images'
    # img_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/images'
    img_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/10000000f620da42/images'
    img_list = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    
    # save_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/output/hue'
    # save_dir = '/home/dorian/Data/cslilcs_2024_october_subsurface_dataset/100000009c23b5af/output/sift'
    save_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/10000000f620da42/output/sift'
    os.makedirs(save_dir, exist_ok=True)
    
    config = {}
    config['denoise_template_window_size'] = 31
    config['denoise_search_window_size'] = 31
    config['denoise_strength'] = 20
    config['min_size'] = 5
    config['max_size'] = 100
    config['contrast_threshold'] = 0.015
    config['edge_threshold'] = 100
    config['sigma'] = 2.0 # 30 is too much, 2.0 so far seems good
    config['dilate'] = 45
                 
    sift = FilterSift(config=config)
    max_img = 10
    for i, img_name in enumerate(img_list):
        print()
        print(f'{i}: {img_name}')
        if i >= max_img:
            print('reached max image limit')
            break
        img_bgr = cv.imread(img_name)
        kp = sift.get_best_sift_features(img_bgr)
        
        # check:
        # sift.print_keypoints(kp)
        
        # draw
        img_ftr = sift.draw_keypoints(img_bgr, kp)
        
        # save
        sift.save_image(img_ftr, img_name, save_dir, '_sift.jpg')
    
        # draw mask of sift regions
        mask_sift = sift.create_sift_mask(img_bgr, kp)
        mask_overlay = sift.display_mask_overlay(img_bgr, mask_sift)
        sift.save_image(mask_overlay, img_name, save_dir, '_siftoverlay.jpg')
        
        
    print('done')
        
        
        
    # import code
    # code.interact(local=dict(globals(), **locals()))
        
        
        