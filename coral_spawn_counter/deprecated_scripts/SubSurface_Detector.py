#! /usr/bin/env python3

"""
coral spawn counting using blobs (copied from count_coral_spawn.py)
TODO: also calls yolov8 detector onto cslics surface embryogenesis
"""
import os
import cv2 as cv
import numpy as np
import glob
# import random as rng
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib as mpl
# import seaborn.objects as so
import pickle
# import pandas as pd
# from datetime import datetime
import sys
sys.path.insert(0, '')
import time
import torch
import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image as MvtImage
from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.Detector import Detector


class SubSurface_Detector(Detector):

    DEFAULT_META_DIR = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    DEFAULT_IMG_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/subsurface'
    DEFAULT_SAVE_DIR = '/home/cslics04/images/subsurface'
    
    DEFAULT_OUTPUT_FILE = 'subsurface_detections.pkl'
    DEFAULT_WINDOW_SIZE = 20
    MAX_IMG = 100000

    # as percentages of the image width
    DEFAULT_P_K_BLUR = 0.0046 # 11
    DEFAULT_P_K_FOCUS = 0.0039
    DEFAULT_P_K_MORPH = 0.0025 # 21 - for hi-res
    
    # as percentage of total area
    DEFAULT_AREA_MIN  = 0.000244
    DEFAULT_AREA_MAX = 0.00325
    DEFAULT_CIRC_MIN = 0.5
    DEFAULT_CIRC_MAX = 1.1
    
    DEFAULT_IMAGE_SIZE = 1280 # resize image to this size, to ensure consistent kernel values
    
    # assuming 1/5 seconds img capture rate
    # 1 img/10 seconds = 2
    # 1 img/15 seconds = 3
    # 1 img/20 seconds = 4
    # 1 img/30 seconds = 6
    DEFAULT_SKIP_IMG = 6

    def __init__(self,
                 meta_dir: str = DEFAULT_META_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_img: int = MAX_IMG,
                 save_prelim_img: bool = True,
                 img_size: int = DEFAULT_IMAGE_SIZE,
                 detection_file: str = DEFAULT_OUTPUT_FILE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 p_area_min: float = DEFAULT_AREA_MIN,
                 p_area_max: float = DEFAULT_AREA_MAX,
                 circ_min: float = DEFAULT_CIRC_MIN,
                 circ_max: float = DEFAULT_CIRC_MAX,
                 skip_img: int = DEFAULT_SKIP_IMG,
                 k_blur: float = DEFAULT_P_K_BLUR,
                 k_morph: float = DEFAULT_P_K_MORPH,
                 k_focus: float = DEFAULT_P_K_FOCUS):
        
        self.detection_file = detection_file
        self.window_size = window_size
        self.capture_time_list = []
        #help with detection
        self.count = 0
        self.blobby = []
        self.icont = []
        
        # detection parameters for finding edges/creating blobs
        self.p_k_blur = k_blur
        self.p_k_morph = k_morph
        self.p_k_focus = k_focus
        self.p_k_focus_blur = k_blur
        
        # parameters for circle/blob thresholding/filtering:
        self.p_area_min = p_area_min
        self.p_area_max = p_area_max
        self.circ_min = circ_min
        self.circ_max = circ_max

        # parameters for plotting subsurface detections
        self.contour_colour = [255, 0, 0] # red in BGR
        self.contour_thickness = 4
        
        self.skip_img = skip_img
        
        Detector.__init__(self, 
                          meta_dir = meta_dir,
                          img_dir = img_dir,
                          save_dir = save_dir,
                          max_img = max_img,
                          save_prelim_img = save_prelim_img,
                          img_size=img_size)

    def make_odd(self, num):
        if num %2 == 0:
            return num + 1
        else:
            return num
    
    def resize_image(self, image, desired_width):
        # resize image according to desired width, preserving the aspect ratio
        aspect_ratio = desired_width / image.shape[1]
        height = int(image.shape[0] * aspect_ratio)
        resized_image = cv.resize(image, (desired_width, height), interpolation=cv.INTER_AREA)
        return resized_image
        
    def adjust_kernels_to_image_size(self, img):
        # adjust kernels to image size, given as a percentage
        # ensure they are odd and int
        # image size may change over the course of the operation
        [h, w, c] = img.shape
        self.k_blur = self.make_odd(int(self.p_k_blur * w))
        self.k_morph = self.make_odd(int(self.p_k_morph * w))
        self.k_focus = self.make_odd(int(self.p_k_focus * w))
        # self.k_focus = 5 # TODO make percentage
        self.k_focus_blur = self.make_odd(int(self.p_k_focus_blur * w))
        
        a = w * h
        self.area_min = int(self.p_area_min * a)
        self.area_max = int(self.p_area_max * a)
        
        print(f'kernel blur = {self.k_blur}')
        print(f'kernel morph = {self.k_morph}')
        print(f'kernel focus = {self.k_focus}')
        print(f'kernel focus blur = {self.k_focus_blur}')
        
        # import code
        # code.interact(local=dict(globals(), **locals()))
            
    def prep_img_name(self, img_name):
        # create coral image:
        # TODO might need to change txt_name to actual img_name? or it's just a method of getting the capture_time metadata
        # coral_image = CoralImage(img_name=img_name, txt_name = 'placeholder.txt')
        # self.capture_time_list.append(coral_image.metadata['capture_time'])
        
        # read in image
        im = MvtImage(img_name)
        # im = cv.imread(img_name)
        
        return self.prep_img(im, img_name)
        
    def prep_img(self, im, img_name=None):
        """
        from an img_name, load the image into the correct format for dections (greyscaled, blured and morphed)
        """
        im = MvtImage(im)
        filename = im.filename
        # set image size to 1280:
        im = MvtImage(self.resize_image(im.image, self.img_size))
       
        
        # # create coral image:
        # # TODO might need to change txt_name to actual img_name? or it's just a method of getting the capture_time metadata
        capture_date = os.path.basename(filename).split('_')[1]
        capture_time = os.path.basename(filename).split('_')[2]
        self.capture_time_list.append(capture_date + '_' + capture_time)
        
        # set image kernels as a percentage of image size for consistency
        self.adjust_kernels_to_image_size(im)
        
        #process image
        # TODO confusingly, we switch between opencv format and MVT format, should choose one and stick with
        im_mono = cv.cvtColor(im.image, cv.COLOR_BGR2GRAY) # grayscale 
        im_blur = cv.GaussianBlur(im_mono, (self.k_blur, self.k_blur), 0)         # blur image
        
        # try applying non-local means noising to smooth out flat sections, but preserve edges
        im_blur = cv.fastNlMeansDenoising(im_blur, None, 10, 7 ,21)
        
        # updated method: Laplacian, which shows edges without as much sensitivity to initial thresholds
        # although the method calls the sobel operator under the hood, it produces a much more consistent grayscale image
        # that can then be more reliably thresholded wrt edges (ie, "in-focus corals")
        focus_measure = cv.Laplacian(im_blur, cv.CV_16S, ksize=self.k_focus)
        im_focus = cv.convertScaleAbs(focus_measure)
        # im_focus = MvtImage(im_focus)
        # smooth focus image
        im_focus = cv.GaussianBlur(im_focus, (self.k_focus_blur, self.k_focus_blur), 0)
        
        # TODO - create histogram of image! - save it for analysis
        
        # threshold image
        # simple threhsolding
        # thresh = 100
        # thresh, im_thresh = cv.threshold(im_focus, thresh, maxval=255, type=cv.THRESH_BINARY)
        # otsu's / triangle
        # thresh, im_thresh = cv.threshold(im_focus, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        thresh, im_thresh = cv.threshold(im_focus, 0, 255, cv.THRESH_BINARY+cv.THRESH_TRIANGLE)
        # adaptive thresholding
        # im_thresh = cv.adaptiveThreshold(im_focus.image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,61, 2)
        # save_thresh_img_name = os.path.join(save_dir, img_base_name + '_02_thresh.jpg') 
            
        # morph
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(self.k_morph,self.k_morph))
        # kernel = np.ones((k_morph,k_morph), np.uint8)
        
        # TODO remove MvtImage dependency
        im_thresh = MvtImage(im_thresh)
        im_morph = MvtImage(im_thresh.dilate(kernel, n=1)) # was erode...?
        # im_morph = MvtImage(im_morph.erode(kernel, n=1))
        im_morph = im_morph.open(kernel)
        im_morph = im_morph.close(kernel)

        # step-by-step save the image
        if self.save_prelim_img:
            
            # img_base_name = os.path.basename(img_name)[:-4]
            img_base_name = os.path.basename(img_name)[:-4]
            cv.imwrite(os.path.join(self.save_dir, img_base_name + '_00_orig.jpg'), im_mono)
            cv.imwrite(os.path.join(self.save_dir, img_base_name + '_01_blur.jpg'), im_blur)
            # im_canny.write(os.path.join(self.save_dir, img_base_name + '_02_edge.jpg'))
            cv.imwrite(os.path.join(self.save_dir, img_base_name + '_02_focus.jpg'), im_focus)
            # TODO remove MvtImage dependency
            im_thresh.write(os.path.join(self.save_dir, img_base_name + '_03_thresh.jpg'))
            im_morph.write(os.path.join(self.save_dir, img_base_name + '_04_morph.jpg'))
        
        # be sure to return a numpy array, just like other Detectors
        return im_morph.image


    def attempt_blobs(self, image, img_name: str=None): # , original_image):
        """
        given an img_name, a morphed image and the original image,
        try to find blobs in the image and threashold these blobs
        """
        try:
            
            # img_base_name = os.path.basename(img_name)[:-4]
            image = MvtImage(image)
            blobby = mvt.Blob(image)
            print(f'{len(blobby)} blobs initially found')
            # show blobs
            # import code
            # code.interact(local=dict(globals(), **locals()))
            if self.save_prelim_img:
                imblobs = blobby.drawBlobs(image, None, None, None, contourthickness=-1)
            
            # reject too-small and too weird (non-circular) blobs
            b0 = [b for b in blobby if ((b.area < self.area_max and b.area > self.area_min) and (b.circularity > self.circ_min and b.circularity < self.circ_max))]
            b0_area = [b.area for b in b0]
            b0_circ = [b.circularity for b in b0]
            b0_cent = [b.centroid for b in b0]
            # get index of blobbs that passed thresholds
            icont = [i for i, b in enumerate(blobby) if (blobby[i].centroid in b0_cent and 
                                                            blobby[i].area in b0_area and 
                                                            blobby[i].circularity in b0_circ)] 
            
            # print('icont')
            if self.save_prelim_img:
                imblobs_area = blobby.drawBlobs(image, None, icont, None, contourthickness=-1)

                # plot the contours of the blobs based on imblobs_area and icont:
                image_contours = image.image
                
                # if 2D image (no colour), make it 3D
                if len(image_contours.shape) == 2:
                    image_contours = cv.cvtColor(image_contours, cv.COLOR_GRAY2RGB)

                # print('drawing contours')
                for i in icont:
                    
                    cv.drawContours(image_contours,
                                    blobby._contours,
                                    i,
                                    self.contour_colour,
                                    thickness=self.contour_thickness,
                                    lineType=cv.LINE_8)
                image3 = MvtImage(image_contours)
                if img_name is None:
                    img_base_name = 'blob_image'
                else:
                    img_base_name = os.path.basename(img_name)
                

                # print('saving in-progress images')
                image3.write(os.path.join(self.save_dir, img_base_name + '_blobs_contour.jpg'))

                imblobs.write(os.path.join(self.save_dir, img_base_name + '_04_blob.jpg'))
                imblobs_area.write(os.path.join(self.save_dir, img_base_name + '_05_blob_filter.jpg'))
            print(f'{len(icont)} blobs passed thresholds')
            return blobby, icont
        except:
            print(f'Try/Except: No blob detection for image')
            # # append empty to keep indexes consistent
            return [], []


    def save_results_2_pkl(self, pkl_save_path, blobs_list, blobs_count, image_index, capture_time):
        """
        saving blobs list, blobs count and other data to a pickle file
        """
        with open(pkl_save_path, 'wb') as f:
            save_data ={'blobs_list': blobs_list,
                'blobs_count': blobs_count,
                'image_index': image_index,
                'capture_time': capture_time}
            pickle.dump(save_data, f)


    def blob_2_box(self, img, icont, blobby):
        """
        convert the blobs into bounding boxes of yolo format ([x1 y1 x2 y2 conf class_idx class_name])
        only saving the blobs that passed the threashold in attempt_blobs
        """
        # img = cv.imread(img_name)
        if icont == []: #incase of no blobs
            return []
        contours = blobby._contours
        imgw, imgh = img.shape[1], img.shape[0]
        bb = []
        for i, c in enumerate(contours):
            if i in icont:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
                x1 = x/imgw
                y1 = y/imgh
                x2 = (x+w)/imgw
                y2 = (y+h)/imgh
                bb.append([x1, y1, x2, y2, 0.5, 3, 3])
        #debug visualise
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.show()
        return torch.tensor(bb)


    def detect(self, image: np.ndarray):
        """
        return detections from a single prepared, binary image, 
        attempts to find blobs and then converts the blobs into yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        # img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        # img_name = img_list[self.count]
        # blobby, icont = self.attempt_blobs(img_name, image, im = MvtImage(img_name))
        blobby, icont = self.attempt_blobs(image)
        predictions = self.blob_2_box(image, icont, blobby)
        self.blobby = blobby
        self.icont = icont
        self.count += 1
        return predictions
    
    
    def run(self):
        
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg'))) # use `*/*.jpg` when img_dir is sub-divided by date folders
        print(f'running blob subsurface detector on {len(img_list)} images')
        
        imgsave_dir = os.path.join(self.save_dir, 'detection_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detection_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        blobs_count = []
        blobs_list = []
        image_index = []

        start_time = time.time()

        for i, img_name in enumerate(img_list):
            if i >= self.max_img:
                print(f'{i}: hit max img - stop here')
                break
    
            # skip_interval = 2
            if i % self.skip_img == 0: # if even
                print(f'predictions on {i+1}/{len(img_list)}')
                im_morph = self.prep_img_name(img_name) 
                print('image prepared')
                image_index.append(i)

                predictions = self.detect(im_morph)
                #self.save_image_predictions(predictions, MvtImage(img_name).image, img_name, imgsave_dir, BGR=False)
                #self.save_text_predictions(predictions, img_name, txtsavedir)
                
                blobs_list.append(self.blobby)
                blobs_count.append(self.icont)
        
        pkl_save_path = os.path.join(self.save_dir, self.detection_file)
        self.save_results_2_pkl(pkl_save_path, blobs_list, blobs_count, image_index, self.capture_time_list) 

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))

        print(f'time[s]/image = {duration / len(img_list)}')
        
        # capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in self.capture_time_list]
        # decimal_days = convert_to_decimal_days(capture_time_dt)

        # return decimal_days

    
def main():

    # root_dir = '/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics01'
    # meta_dir = '/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics01'
    # img_dir = '/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics01/subsurface_test'
    # save_dir = '/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics01/test_output'
    MAX_IMG = 999999999999999999999
    detection_file = 'test_subsurface_detections.pkl'
    img_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06/images'
    save_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06/detections_subsurface'
    meta_dir = '/home/java/Java/cslics' 
    Coral_Detector = SubSurface_Detector(meta_dir = meta_dir, 
                                         img_dir = img_dir, 
                                         save_dir=save_dir, 
                                         detection_file=detection_file,
                                         max_img = MAX_IMG, 
                                         skip_img=100)
    Coral_Detector.run()
    import code
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()

