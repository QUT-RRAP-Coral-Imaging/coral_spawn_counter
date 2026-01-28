#!/usr/bin/env python3

""" base class for detectors
has detector.detect()
has detector.save_image_predictions
has detector.save_to_text_predictions
set class colours, etc
"""


import os
import cv2 as cv
import re
# import numpy as np
import glob as glob
# from datetime import datetime


class Detector(object):
    
    DEFAULT_META_DIR = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    DEFAULT_IMG_DIR = '/home/cslics04/20231018_cslics_detector_images_sample/microspheres'
    DEFAULT_SAVE_DIR = '/home/cslics04/images/redcircles'
    
    DEFAULT_MAX_IMG = 15
    SUFFIX_TXT = '_det.txt'
    SUFFIX_IMG = '_det.jpg'
    
    def __init__(self, 
                 meta_dir: str = DEFAULT_META_DIR,
                 img_dir: str = DEFAULT_IMG_DIR,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 max_img: int = DEFAULT_MAX_IMG,
                 save_prelim_img: bool = False,
                 img_size: int = 1280):
        
        # folders and files
        self.meta_dir = meta_dir
        self.img_dir = img_dir
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')) +
                               glob.glob(os.path.join(self.img_dir, '*.png')) + 
                               glob.glob(os.path.join(self.img_dir, '*.jpeg')))
        
        
        self.save_dir = save_dir
        print(f'Detector.save_dir = {self.save_dir}')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.classes = self.get_classes(self.meta_dir)
        self.class_colours = self.set_class_colours(self.classes)
        
        
        # common parameters for detectors
        self.max_img = max_img
        self.save_prelim_img = save_prelim_img
        self.img_size = img_size
        
        
    def prep_img_name(self, img_name: str):
        """
        from an img name, load the image into the correct format for dections (rgb)
        """
        img_bgr = cv.imread(img_name) # BGR
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # RGB
        return img_rgb


    def get_classes(self, meta_dir):
        """
        get the classes from a metadata/obj.names file
        classes = [class1, class2, class3 etc.]
        """
        with open(os.path.join(meta_dir, 'metadata','obj.names'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    
    def set_class_colours(self, classes):
        """
        set classes to specific colours using a dictionary
        """
        #TODO: make a function of something else, used in both detectors
        orange = [255, 128, 0] # four-eight cell stage
        blue = [0, 212, 255] # first cleavage
        purple = [170, 0, 255] # two-cell stage
        yellow = [255, 255, 0] # advanced
        brown = [144, 65, 2] # damaged
        green = [0, 255, 00] # egg
        colors = [orange, blue, purple, yellow, brown, green]
        # iterate on length/number of classes (for flexibility, in case number of classes changes)
        class_colours = {}
        for i, c in enumerate(classes):
            # TODO incrementall add to the dictionary
            class_colours.update({c: colors[i]})
            
        # class_colours = {classes[0]: orange,
        #                 classes[1]: blue,
        #                 classes[2]: purple,
        #                 classes[3]: yellow,
        #                 classes[4]: brown,
        #                 classes[5]: green}
        return class_colours

    def get_date_str(self, relative_filename, pattern=None):
        """get_date_str
        gets date string from filename following cslics image naming convention
        designed for basename only, untested with absolute filepaths
        Args:
            name (_type_): _description_
        """
        if pattern is None:
            pattern = r'cslics\d+_(\d{8})'
            # regular expression to extract the date from the filename of the format:
            # 'cslics01_20231205_234344_838548_img.jpg'
        match = re.search(pattern, relative_filename)
        if match:
            date_str = match.group(1)
        else:
            date_str = None
            print('get_date_str: no date found in filename')
        return date_str
        

    def save_text_predictions(self, predictions, imgname, txtsavedir):
        """
        save predictions/detections into text file
        [x1 y1 x2 y2 conf class_idx class_name]
        """
        txtsavename = os.path.basename(imgname)
        date_str = self.get_date_str(txtsavename)
        os.makedirs(os.path.join(txtsavedir, date_str), exist_ok=True)
        
        txtsavepath = os.path.join(txtsavedir, date_str, txtsavename[:-4] + self.SUFFIX_TXT)

        # predictions [ pix pix pix pix conf class ]
        with open(txtsavepath, 'w') as f:
            for p in predictions:
                x1, y1, x2, y2 = p[0:4].tolist()
                conf = p[4]
                class_idx = int(p[5])
                class_name = self.classes[class_idx]
                f.write(f'{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.4f} {class_idx:g} {class_name}\n')
        return True
    
    
    def save_image_predictions(self, predictions, img, imgname, imgsavedir, BGR=False, quality=50):
        """
        save predictions/detections (assuming predictions in yolo format) on image
        """
        # img = cv.imread(imgname)
        # assuming input image is rgb, need to convert back to bgr:
        
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            #extract back into cv lengths
            x1 = x1*imgw
            x2 = x2*imgw
            y1 = y1*imgh
            y2 = y2*imgh        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)

        imgsavename = os.path.basename(imgname)
        # add day into save directory to prevent an untenable number of images in a single folder
        # get date from imgsavename
        date_str = self.get_date_str(imgsavename)
        os.makedirs(os.path.join(imgsavedir, date_str), exist_ok=True)
        
        imgsave_path = os.path.join(imgsavedir, date_str, imgsavename.rsplit('.',1)[0] + self.SUFFIX_IMG)
        
        if BGR:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)        
        # to save on memory, reduce quality of saved image
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)
        return True

        
    def convert_to_decimal_days(self, dates_list, time_zero=None):
        if time_zero is None:
            time_zero = dates_list[0]  # Time zero is the first element date in the list
        else:
            time_zero = time_zero
            
        decimal_days_list = []

        for date in dates_list:
            time_difference = date - time_zero
            decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
            decimal_days_list.append(decimal_days)

        return decimal_days_list


if __name__ == "__main__":

    print('Detector.py')
    meta_dir = '/home/cslics/Code/cslics_ws/coral_spawn_imager'
    img_dir = '/home/cslics/Data/20231018_cslics_detector_images_sample/microspheres'
    save_dir = '/home/cslics/Data/20231018_cslics_detector_images_sample/detections'
    Det = Detector(meta_dir=meta_dir, img_dir=img_dir, save_dir=save_dir)
    # test null case for printing/saving images
    p = []

    img_name = Det.img_list[0]
    img = Det.prep_img_name(img_name)

    print('save_text_predictions')
    Det.save_text_predictions(p,img_name, save_dir, Det.classes)

    print('save_image_predictions')
    Det.save_image_predictions(p, img, img_name, save_dir,Det.class_colours,Det.classes)



