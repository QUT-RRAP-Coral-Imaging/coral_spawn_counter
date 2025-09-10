#! /usr/bin/env python3

"""
Image class, that (TODO: will import from MVT)
but for now, just has metadata reader/extracter
properties for counts, detections
"""

# import PIL.Image as PIL_Image
# from PIL.PngImagePlugin import PngInfo
import json
import numpy as np
import os
from pprint import *

from annotation.Image import Image
from coral_spawn_counter.CircleDetector import CircleDetector

class CoralImage(Image):

    def __init__(self, img_name, txt_name, img=None, detections=None, fertratio=None):
        
        Image.__init__(self, img_name)
        # for now, only jpgs or pngs
        self.img_name = img_name
        self.img = None # PIL_Image.open(img_name) # insane memory costs for large number of Images
        self.txt_name = txt_name
        self.count = 0
        self.detections = detections
        self.fertratio = fertratio
        self.SpawnCounter = CircleDetector() # TODO push this into separate function


    def count_spawn(self, img, det_param=None):    # TODO make this a unique function for circle detection
        count, circles = self.SpawnCounter.count_spawn(np.array(img), det_param)
        self.count = count
        self.detections = circles


    def save_detection_img(self, img, img_name=None, save_dir=None, resize=0.5):
        if img_name is None:
            img_name = self.img_dir
        if save_dir is None:
            path = os.path.dirname(__file__)
            save_dir = os.path.join(path, 'detections')
        
        # print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.SpawnCounter.save_detections(img=np.array(img), img_name=img_name, save_dir=save_dir, resize=resize)


    # def set_img(self, img):
    #     self.img = img

    def set_count(self, count):
        self.count = count

    def set_detections(self, detections):
        self.detections = detections
    
    def get_count(self):
        return self.count

    def get_detections(self):
        return self.detections

    def numpy_array(self):
        return np.array(self.img)



if __name__ == "__main__":

    print('CoralImage.py')

    img_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data/cslics03/images'
    img_list = os.listdir(img_dir)
    img_list.sort()
    img = CoralImage(os.path.join(img_dir, img_list[-1]))
    img.read_metadata(img_name = os.path.join(img_dir, img_list[4]))


    import code
    code.interact(local=dict(globals(), **locals()))