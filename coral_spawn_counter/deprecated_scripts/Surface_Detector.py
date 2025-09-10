#! /usr/bin/env python3

"""
use the trained yolov5 model, and run it on a given folder/sequence of images
runs standard detection of cslics surface, but can also compare groundtruth with predictions
"""

import os
import torch
import torchvision
import glob
# import numpy as np
# from PIL import Image as PILImage
import cv2 as cv
import time
import pickle
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import sys
from datetime import datetime
sys.path.insert(0, '')
from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.Detector import Detector

import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile

class Surface_Detector(Detector):

    ROOT_DIR = '/home/dorian' # '/home/cslics04
    DEFAULT_META_DIR = os.path.join(ROOT_DIR, '/cslics_ws/src/coral_spawn_imager')
    # DEFAULT_IMG_DIR = os.path.join(ROOT_DIR,'/Data/20231018_cslics_detector_images_sample/surface')
    DEFAULT_IMG_DIR = os.path.join('/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics01/images')
    DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR,'/images/surface')
    DEFAULT_TXT_DIR = os.path.join(ROOT_DIR,'/images/surface_txt')
    DEFAULT_TIME = None # minutes
    
    DEFAULT_DETECTOR_IMAGE_SIZE = 1280
    DEFAULT_CONFIDENCE_THRESHOLD = 0.30
    DEFAULT_IOU = 0.5
    DEFAULT_MAX_IMG = 100000 # per image
    DEFAULT_MAX_DET = 999 # max detections per image
    
    DEFAULT_IMG_PATTERN = '*/*.jpg' # default image pattern for finding images/types of images. '*/*.jpg' looks for one sub-level of folders (date) and jpgs
    
    # yolo path on cslics unit
    DEFAULT_YOLO8 = os.path.join('/home/cslics04/cslics_ws/src/ultralytics_cslics/weights', 'cslics_20230905_yolov8n_640p_amtenuis1000.pt')
    # yolo path on dorian's computer
    # DEFAULT_YOLO8 = os.path.join( '/home/dorian/Code/cslics_ws/src/ultralytics_cslics/weights','cslics_20240117_yolov8x_640p_amt_alor2000.pt')
    DEFAULT_OUTPUT_FILE = 'surface_detections.pkl'
    DEFAULT_SKIP_INTERVAL = 1
                 
    DEFAULT_SAVE_IMAGE = True
    DEFAULT_SAVE_TEXT = True
    
                 
    def __init__(self,
                meta_dir: str = DEFAULT_META_DIR,
                img_dir: str = DEFAULT_IMG_DIR,
                save_dir: str = DEFAULT_SAVE_DIR,
                max_img: int = DEFAULT_MAX_IMG,
                img_size: int = DEFAULT_DETECTOR_IMAGE_SIZE,
                weights_file: str = DEFAULT_YOLO8,
                conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
                iou: float = DEFAULT_IOU,
                output_file: str = DEFAULT_OUTPUT_FILE,
                txt_dir: str = DEFAULT_TXT_DIR,
                skip_img: int = DEFAULT_SKIP_INTERVAL,
                max_det: int = DEFAULT_MAX_DET,
                img_pattern: str = DEFAULT_IMG_PATTERN,
                time_lim: int = DEFAULT_TIME,
                save_img: bool = DEFAULT_SAVE_IMAGE,
                save_txt: bool = DEFAULT_SAVE_TEXT):
        
        self.weights_file = weights_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if not torch.cuda.is_available():
            print('Note: torch cuda (GPU) is not available, so performance will be slower')
            
        self.model = YOLO(weights_file)
        self.conf = conf_thresh
        self.iou = iou
        self.max_det = max_det
        self.img_pattern = img_pattern
        self.time_lim = time_lim

        self.save_img = save_img
        self.save_txt = save_txt
        self.output_file = output_file
        self.txt_dir = txt_dir
        self.skip_interval = skip_img
        Detector.__init__(self, 
                          meta_dir = meta_dir,
                          img_dir = img_dir,
                          save_dir = save_dir,
                          max_img = max_img,
                          img_size=img_size)


    def nms(self, pred, conf_thresh, iou_thresh):
        """ perform class-agnostic non-maxima suppression on predictions 
        pred = [x1 y1 x2 y2 conf class] tensor
        # TODO handle the pred empty case!
        """

        # Checks
        assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0'
        # if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        #     prediction = prediction[0]  # select only inference output

        # conf = object confidence * class_confidence
        pred = pred[pred[:, 4] > conf_thresh]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        # keep = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)

        # sort scores into descending order
        _, indices = torch.sort(scores, descending=True)

        # class-agnostic nms
        # iteratively adds the highest scoring detection to the list of final detections, 
        # calculates the IoU of this detection with all other detections, and 
        # removes any detections with IoU above the threshold. 
        # This process continues until there are no detections left.
        keep = []
        while indices.numel() > 0:
            # get the highest scoring detection
            i = indices[0]
            
            # add the detection to the list of final detections
            keep.append(i.item())

            # calculate the IoU highest scoring detection within all other detections
            if indices.numel() == 1:
                break
            else:
                overlaps = torchvision.ops.box_iou(boxes[indices[1:]], boxes[i].unsqueeze(0)).squeeze()

            # keep only detections with IOU below certain threshold
            indices = indices[1:]
            indices = indices[overlaps <= iou_thresh]

        return pred[keep, :]


    def convert_results_2_pkl(self, txtsavedir, save_file_name, imglist):
        """
        assuming detections have already been done (i.e. txtsavedir is full of txtfiles)
        convert textfiles and images into CoralImage and save data in pkl file,
        assuming textfiles in [[x1 y1 x2 y2 conf class_idx class_name] yolo formt
        """
        # read in each .txt file
        # txt_list = sorted(os.listdir(txtsavedir))
        txt_list = sorted(glob.glob(os.path.join(txtsavedir, '*.txt')))
        if len(txt_list) > 0:
            results = []
            for i, txt in enumerate(txt_list):
                print(f'importing detections {i+1}/{len(txt_list)}')
                txt_basename = os.path.basename(txt)
                with open(os.path.join(txtsavedir, txt_basename), 'r') as f:
                    detections = f.readlines() # [x1 y1 x2 y2 conf class_idx class_name] \n
                detections = [det.rsplit() for det in detections]

                # corresponding image name:
                folder_name = imglist[i*self.skip_interval].split('/')[-2]
                #folder_name = txt_basename.split('_')[1]
                if self.img_pattern == '*/*.jpg':
                    img_name = os.path.join(self.img_dir, folder_name, txt_basename[:-8] + self.img_pattern[-4:])
                else:
                    img_name = os.path.join(self.img_dir, txt_basename[:-8] + self.img_pattern[-4:])
                try:
                    CImage = CoralImage(img_name=img_name, # TODO absolute vs relative? # want to grab the metadata
                                        txt_name=txt,
                                        detections=detections)
                    results.append(CImage)
                except:
                    print(f'problem with {txt_basename}, when saving as CImage')
                    import code
                    code.interact(local=dict(globals(), **locals()))
                
            # sort results based on metadata capture time
            results.sort(key=lambda x: x.metadata['capture_time'])

            savefile = os.path.join(self.save_dir, save_file_name)
            with open(savefile, 'wb') as f:
                pickle.dump(results, f)
            return True
        else:
            print(f'ERROR: No txt files in {txtsavedir}')
            return False


    def show_image_predictions(self, predictions, img, SHOW=False):
        """
        show predictions/detections (assurming predeictions in yolo format) for an image
        """
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])        
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)
            
        if SHOW:
            cv.show(img)
        return True

    def show_ground_truth(self, img_rgb, imgname, imgsavedir, BGR=True):
        """Shows ground truth annotations on image if they exsit"""
        imgw, imgh = img_rgb.shape[1], img_rgb.shape[0]
        basename = os.path.basename(imgname)
        ground_truth_txt = os.path.join(self.txt_dir, basename[:-4] + '.txt')
        if os.path.exists(ground_truth_txt):
            with open(ground_truth_txt, 'r') as f:
                lines = f.readlines() # <object-class> <x> <y> <width> <height>
            for part in lines:
                parts = part.rsplit()
                class_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                ow = float(parts[3])
                oh = float(parts[4])
                x1 = (x - ow/2)*imgw
                x2 = (x + ow/2)*imgw
                y1 = (y - oh/2)*imgh
                y2 = (y + oh/2)*imgh
                cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[class_idx]], 5)
            imgsave_path = os.path.join(imgsavedir, basename[:-4] + '_tru.jpg')
            if BGR:
                img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)        
            cv.imwrite(imgsave_path, img)
        else:
            print(f'no ground truth annotations for {ground_truth_txt}')

    def draw_dotted_rect(self, img, x1, y1, x2, y2, color, thicknes, gap=30):
        # print(f'x1 = {x1}')
        # import code
        # code.interact(local=dict(globals(), **locals()))
        pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        start=pts[0]
        end=pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            start=end
            end=p
            # draw dashed line
            dist = ((start[0]-end[0])**2 + (start[1]-end[1])**2)**.5
            parts = []
            for i in np.arange(0,dist,gap):
                r = i/dist
                x = int((start[0]*(1-r)+end[0]*r)+.5)
                y = int((start[1]*(1-r)+end[1]*r)+.5)
                p = (x,y)
                parts.append(p)
            for p in parts:
                cv.circle(img,p,thicknes,color,-1)

    def ground_truth_compare_predict(self, img_rgb, imgname, predictions, imgsavedir, BGR=True):
        """Shows an image with ground truth annotations and predictions to help compare the differences"""
        # ground truth section
        imgw, imgh = img_rgb.shape[1], img_rgb.shape[0]
        basename = os.path.basename(imgname)
        ground_truth_txt = os.path.join(self.txt_dir, basename[:-4] + '.txt')
        if os.path.exists(ground_truth_txt):
            with open(ground_truth_txt, 'r') as f:
                lines = f.readlines() # <object-class> <x> <y> <width> <height>
            for part in lines:
                parts = part.rsplit()
                class_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                ow = float(parts[3])
                oh = float(parts[4])
                x1 = (x - ow/2)*imgw
                x2 = (x + ow/2)*imgw
                y1 = (y - oh/2)*imgh
                y2 = (y + oh/2)*imgh
                cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[class_idx]], 4)
        # predictions section
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            #extract back into cv lengths
            x1 = x1*imgw
            x2 = x2*imgw
            y1 = y1*imgh
            y2 = y2*imgh  
            #self.draw_dashed_rect(img_rgb, int(x1), int(y1), int(x2), int(y2), self.class_colours[self.classes[class_idx]], 7)      
            self.draw_dotted_rect(img_rgb, int(x1), int(y1), int(x2), int(y2), self.class_colours[self.classes[cls]], 7)
            #cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], 2)
            cv.putText(img_rgb, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[cls]], 2)
        # save image
        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_cmp.jpg')    
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR) # RGB
        cv.imwrite(imgsave_path, img_bgr)

    def detect(self, image):
        """
        return detections from a single rgb image (numpy array)
        passes prediction through nms and returns them in yolo format [x1 y1 x2 y2 conf class_idx class_name]
        """
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        pred = self.model.predict(source=image,
                                  save=False,
                                  save_txt=False,
                                  save_conf=True,
                                  verbose=False,
                                  imgsz=self.img_size,
                                  conf=self.conf,
                                  iou=self.iou,
                                  max_det=self.max_det)
        boxes: Boxes = pred[0].boxes 
        pred = []
        for b in boxes:
            
            # hope to keep all variables on cuda/GPU for speed
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
            else:
                cls = int(b.cls)
                conf = float(b.conf)
                xyxyn = b.xyxyn.cpu().numpy()[0]
                x1n = xyxyn[0]
                y1n = xyxyn[1]
                x2n = xyxyn[2]
                y2n = xyxyn[3]  
                pred.append([x1n, y1n, x2n, y2n, conf, cls])
        
        # after iterating over boxes, make sure pred is on GPU if available (and a single tensor)
        if torch.cuda.is_available():
            pred = torch.tensor(pred, device="cuda:0")
        else:
            pred = torch.tensor(pred)
            
        # TODO should handle empty case!
        if len(pred) > 0:
            predictions = self.nms(pred, self.conf, self.iou)
        else:
            predictions = [] # empty/0 case
        return predictions
    

    def run(self):
        print('running Surface_Detector.py on:')
        print(f'source images: {self.img_dir}')
        imglist = sorted(glob.glob(os.path.join(self.img_dir, self.img_pattern))) # NOTE added for images/DATE/*.jpg
        
        # where to save image and text detections
        imgsave_dir = os.path.join(self.save_dir, 'detection_images')
        os.makedirs(imgsave_dir, exist_ok=True)
        txtsavedir = os.path.join(self.save_dir, 'detection_textfiles')
        os.makedirs(txtsavedir, exist_ok=True)

        start_time = time.time()
        
        if self.time_lim is not None:
            datetime0 = datetime.strptime(imglist[0].split('_')[-4]+imglist[0].split('_')[-3], "%Y%m%d%H%M%S")
            print(f'time limit set to {self.time_lim} minutes, first img date: {datetime0}')

        for i, imgname in enumerate(imglist):
            # SKIP every 2 images to save time:
            skip_interval = self.skip_interval
            # print(f'skipping every {skip_interval} images')
            if i % skip_interval == 0: # if even
                    
                print(f'predictions on {imgname}, processing {i+1}/{len(imglist)}')
                if i >= self.max_img: # for debugging purposes
                    print(f'hit max_img: {self.max_img}')
                    break

                #time lim for fertilisation
                if self.time_lim is not None:
                    current_datetime = datetime.strptime(imglist[i].split('_')[-4]+imglist[i].split('_')[-3], "%Y%m%d%H%M%S")
                    time_difference_minutes = (current_datetime - datetime0).total_seconds() / 60
                    if time_difference_minutes > self.time_lim:
                        print(f'time limit reached: {time_difference_minutes} minutes')
                        break
                # load image
                try:
                    img_rgb = self.prep_img_name(imgname)
                    imageCopy = img_rgb.copy()
                except:
                    print('unable to read image --> skipping')
                    predictions = []
                
                try: 
                    predictions = self.detect(img_rgb)# inference
                except:
                    print('no model predictions --> skipping')
                    predictions = [] # this shouldn't happen, so enter debug mode if it does
                    import code
                    code.interact(local=dict(globals(), **locals()))

                # save predictions as an image
                if self.save_img:
                    self.save_image_predictions(predictions, img_rgb, imgname, imgsave_dir, BGR=True)
                # save predictions as a text file
                if self.save_txt:
                    self.save_text_predictions(predictions, imgname, txtsavedir)
                #self.ground_truth_compare_predict(img_rgb, imgname, predictions, imgsave_dir)
                #self.show_ground_truth(imageCopy, imgname, imgsave_dir)
            

        end_time = time.time()
        duration = end_time - start_time
        print('run time: {} sec'.format(duration))
        print('run time: {} min'.format(duration / 60.0))
        print('run time: {} hrs'.format(duration / 3600.0))
        
        print(f'time[s]/image = {duration / len(imglist)}')
        
        print('done detection')
        self.convert_results_2_pkl(txtsavedir, self.output_file, imglist)
        print(f'results stored in {self.output_file} file')

def to_XML(base_file, img_location, output_file, classes, Coral_Detector):

    tree = ET.parse(base_file)
    root = tree.getroot() 
    new_tree = ElementTree(Element("annotations"))
    # add version element
    version_element = ET.Element('version')
    version_element.text = '1.1'
    new_tree.getroot().append(version_element)
    # add Meta elements, (copy over from source_file)
    meta_element = root.find('.//meta')
    if meta_element is not None:
        new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')
        # copy all subelements of meta
        for sub_element in meta_element:
            new_meta_elem.append(sub_element)
        
    for i, image_element in enumerate(root.findall('.//image')):
        print(i,'images being processed')
        image_id = image_element.get('id')
        image_name = image_element.get('name')
        image_width = int(image_element.get('width'))
        image_height = int(image_element.get('height'))

        # create new image element in new XML
        new_elem = SubElement(new_tree.getroot(), 'image')
        new_elem.set('id', image_id)
        new_elem.set('name', image_name)
        new_elem.set('width', str(image_width))
        new_elem.set('height', str(image_height))
        
        image_file = os.path.join(img_location, image_name)
        img_bgr = cv.imread(image_file)
        
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        predictions = Coral_Detector.detect(img_rgb)

        for j, p in enumerate(predictions):
            try: 
                xyxy = p[0:4].tolist()
                label = classes[int(p[5].item())]
                xtl = min(xyxy[0],xyxy[2])*image_width
                xbr = max(xyxy[0],xyxy[2])*image_width
                ytl = min(xyxy[1],xyxy[3])*image_height
                ybr = max(xyxy[1],xyxy[3])*image_height
                box_elem = SubElement(new_elem, 'box')
                box_elem.set('label', label)
                box_elem.set('source', 'semi-auto')
                box_elem.set('occluded', '0')
                box_elem.set('xtl', str(xtl))
                box_elem.set('ytl', str(ytl))
                box_elem.set('xbr', str(xbr))
                box_elem.set('ybr', str(ybr))
                box_elem.set('z_order', '0')
            except:
                print(f'prediction {j} encountered problem p = {p}')
                import code
                code.interact(local=dict(globals(), **locals()))
            #import code
            #code.interact(local=dict(globals(), **locals()))

    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)

    zip_filename = output_file.split('.')[0] + '.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname='output_xml_file.xml')
    print('XML file zipped')

    # import code
    # code.interact(local=dict(globals(), **locals()))

def main():
    ######### For testing
    # weights = '/home/java/Java/ultralytics/runs/detect/train - aten_alor_2000/weights/best.pt'
    weights = '/home/dorian/Data/java_data/cslics_surface_detectors_models/cslics_20240116_yolov8m_1280p_amt_alor2000.pt'
    meta_dir ='/home/dorian/Data/cslics_2022_datasets/202211_amtenuis_1000/' # '/home/java/Java/cslics' #has obj.names
    # img_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06/images'
    # save_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06/detect_surface_2000_model'
    save_dir = '/home/dorian/Dropbox/QUT/RRAP/Transition2Deployment/CSLICS/cslics_sample_images/cslics_media_release/coral_spawn'
    img_dir = '/home/dorian/Dropbox/QUT/RRAP/Transition2Deployment/CSLICS/cslics_sample_images/cslics_media_release/'

    Coral_Detector = Surface_Detector(weights_file=weights, meta_dir = meta_dir, img_dir=img_dir, save_dir=save_dir, max_img=99999999999999999999999999)
    Coral_Detector.run()
    # meta_dir = '/home/cslics04/cslics_ws/src/coral_spawn_imager'
    # img_dir = '/home/cslics04/20231018_cslics_detector_images_sample/surface'
    # weights = '/home/cslics04/cslics_ws/src/ultralytics_cslics/weights/cslics_20230905_yolov8n_640p_amtenuis1000.pt'

    #######    For just detection
    # Do detection
    # Coral_Detector = Surface_Detector(weights_file=weights, meta_dir = meta_dir, img_dir=img_dir, max_img=5)
    # meta_dir = '/home/java/Java/cslics' #has obj.names
    # img_dir = '/home/java/Java/data/cslics_aloripedes_n_amtenuis_jan_2000/images/all'
    # weights = '/home/java/Java/cslics/cslics_surface_detectors_models/cslics_20240117_yolov8x_640p_amt_alor2000.pt'
    # save_dir = '/home/java/Java/data/cslics_aloripedes_n_amtenuis_jan_2000/all_dect'
    # Coral_Detector = Surface_Detector(meta_dir=meta_dir, img_dir=img_dir, save_dir=save_dir, weights_file=weights)
    #Coral_Detector.run()

    ######### Human in the loop, convert to cvat xml
    # base_file = "/home/java/Downloads/cslics_alor_n_aten_2000_no_ann/annotations.xml"
    # img_location = img_dir
    # output_filename = "/home/java/Downloads/cslics_aloripedes_n_amtenuis_jan.xml"
    # classes = ["Four-Eight-Cell Stage", "First Cleavage", "Two-Cell Stage", "Advanced Stage", "Damaged", "Egg"]
    # to_XML(base_file, img_location, output_filename, classes, Coral_Detector)

    ########### For comparing prediction with ground truth
    # meta_dir = '/home/java/Java/cslics' #has obj.names
    # img_dir = '/home/java/Java/data/cslics_2924_alor_1000_(2022n23_combined)/images/val'
    # txt_dir = '/home/java/Java/data/cslics_2924_alor_1000_(2022n23_combined)/labels/val'
    # weights = '/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt'
    # save_dir = '/home/java/Java/data/cslics_2924_alor_1000_(2022n23_combined)/detect'
    # Coral_Detector = Surface_Detector(meta_dir=meta_dir, img_dir=img_dir, save_dir=save_dir, weights_file=weights, txt_dir=txt_dir)
    # Coral_Detector.run()

    ################# Cslics desktop
    # meta_dir = '/home/java/Java/cslics' #has obj.names
    # img_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/images'
    # weights = '/home/java/Java/ultralytics/runs/detect/train - alor_atem_1000/weights/best.pt'
    # save_dir = '/home/java/Java/data/cslics_desktop_data/202311_Nov_cslics_desktop_sample_images/detect'
    # base_file = "/home/java/Downloads/cslics_desktop_nov_empty.xml"
    # output_filename = "/home/java/Downloads/cslics_desktop_nov.xml"
    # classes = ["Four-Eight-Cell Stage", "First Cleavage", "Two-Cell Stage", "Advanced Stage", "Damaged", "Egg"]
    # Coral_Detector = Surface_Detector(meta_dir=meta_dir, img_dir=img_dir, save_dir=save_dir, weights_file=weights)
    # to_XML(base_file, img_dir, output_filename, classes, Coral_Detector)
    #Coral_Detector.run()

if __name__ == "__main__":
    main()


# import code
# code.interact(local=dict(globals(), **locals()))


