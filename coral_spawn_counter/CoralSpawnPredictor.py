import os
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import functools
import torch
import cv2 as cv
from ultralytics import YOLO


class CoralSpawnPredictor:
    def __init__(self, weights_path, img_dir, save_dir, classes, class_colours, iou_thresh=0.3, conf_thresh=0.25, max_det=1000, save_img=True, save_txt=True, max_images=None):
        self.weights_path = weights_path
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.classes = classes
        self.class_colours = class_colours
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh  # Add this line
        self.max_det = max_det
        self.save_img = save_img
        self.save_txt = save_txt
        self.max_images = max_images  # Add this line
        self.current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Initialize model
        print(f'Loading model: {weights_path}')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_path).to(self.device)

        # Prepare output directories
        self.imgsave_dir = os.path.join(save_dir, 'detections_images')
        self.txtsave_dir = os.path.join(save_dir, 'detections_txt')
        os.makedirs(self.imgsave_dir, exist_ok=True)
        os.makedirs(self.txtsave_dir, exist_ok=True)


    def save_image_predictions_bb(self, predictions, imgname, imgsavedir):
        """
        Save predictions/detections on image as bounding box.
        """
        FONT_SIZE = 2
        FONT_THICK = 2
        BOX_THICK = 2
        quality = 25

        img = cv.imread(imgname)  # BGR
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            x1, x2 = x1 * imgw, x2 * imgw
            y1, y2 = y1 * imgh, y2 * imgh
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colours[self.classes[cls]], BOX_THICK)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, self.class_colours[self.classes[cls]], FONT_THICK)
        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)


    def save_txt_predictions_bb(self, predictions, imgname, txtsavedir):
        """
        Save predictions/detections as bounding box in text format.
        """
        imgsavename = os.path.basename(imgname)
        txt_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.txt')
        with open(txt_save_path, "w") as file:
            for p in predictions:
                x1, y1, x2, y2 = p[0:4].tolist()
                conf = p[4]
                cls = int(p[5])
                line = f"{x1} {y1} {x2} {y2} {cls} {conf}\n"
                file.write(line)


    def save_json_predictions_bb(self, predictions, imgname, txtsavedir):
        """
        Save predictions as bounding box in JSON format.
        """
        imgsavename = os.path.basename(imgname)
        json_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.json')
        predictions_dict = {
            "model_name": Path(self.weights_path).stem,
            "date run": self.current_datetime,
            "detections [xn1, yn1, xn2, yn2, conf, cls]": predictions.tolist()
        }
        with open(json_save_path, 'w') as f:
            json.dump(predictions_dict, f, indent=4)


    def process_image(self, img_name):
        """
        Process a single image: run inference, save predictions as images, text, and JSON.
        """
        results = self.model.predict(source=img_name, iou=self.iou_thresh, conf=self.conf_thresh, agnostic_nms=True, max_det=self.max_det)
        boxes = results[0].boxes
        pred = []
        for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
        predictions = torch.tensor(pred)

        # Determine relative path for saving
        rel_path = os.path.relpath(os.path.dirname(img_name), self.img_dir)

        # Save image predictions
        if self.save_img:
            os.makedirs(os.path.join(self.imgsave_dir, rel_path), exist_ok=True)
            self.save_image_predictions_bb(predictions, img_name, os.path.join(self.imgsave_dir, rel_path))

        # Save text and JSON predictions
        if self.save_txt:
            os.makedirs(os.path.join(self.txtsave_dir, rel_path), exist_ok=True)
            self.save_txt_predictions_bb(predictions, img_name, os.path.join(self.txtsave_dir, rel_path))
            self.save_json_predictions_bb(predictions, img_name, os.path.join(self.txtsave_dir, rel_path))


    def run(self):
        """
        Run the prediction on all images in the directory.
        """
        print(f'Fetching image list in all subfolders from: {self.img_dir}')
        img_list = sorted(Path(self.img_dir).rglob('*_img.jpg'))
        
        # Apply max_images limit if specified
        if self.max_images is not None and self.max_images > 0:
            img_list = img_list[:self.max_images]
            print(f'Limited to first {self.max_images} images')
        
        print(f'Number of images to process: {len(img_list)}')

        start_time = time.time()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_image, img_list)

        end_time = time.time()
        duration = end_time - start_time

        print('Done')
        print(f'Run time: {duration:.2f} sec')
        print(f'Run time: {duration / 60.0:.2f} min')
        print(f'Run time: {duration / 3600.0:.2f} hrs')
        if len(img_list) > 0:
            print(f'Time[s]/image = {duration / len(img_list):.2f}')
        else:
            print('No images processed.')
        


# Example usage
if __name__ == "__main__":
    # weights_path = '/home/tsaid/data/cslics_datasets/models/cslics_subsurface_20250205_640p_yolov8n.pt'
    # img_dir = '/home/tsaid/data/cslics_datasets/cslics_november_2024/100000000846a7ff'
    # save_dir = '/home/tsaid/data/cslics_datasets/cslics_november_2024/detections/100000000846a7ff'
    
    # weights_path = '/home/dtsai/Data/cslics_datasets/models/cslics_subsurface_202508185_640p_yolov8n.pt'
    # img_dir = '/media/dtsai/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics09/images'
    img_dir = '/media/dtsai/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics08/images'

    save_dir = '/media/dtsai/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/detections/cslics08'

    # format expected by CslicsDataProcessor.py
    # f'{self.base_det_dir}/{self.cslics_uuid}/{self.model_name}'    
    
    # weights_path = '/home/dtsai/Data/cslics_datasets/cslics_2023_subsurface_dataset/models/20241122_cslics_subsurface_yolov8n_640p.pt'
    weights_path = '/home/dtsai/Data/cslics_datasets/models/cslics_subsurface_202508185_640p_yolov8n.pt'
    
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    save_dir = os.path.join(save_dir, model_name)
    # img_dir = '/home/dtsai/Data/cslics_datasets/cslics_2023_subsurface_dataset/runs/combined/split/images/train'
    # save_dir = '/home/dtsai/Data/cslics_datasets/cslics_2023_subsurface_dataset/runs/combined/detections'
    classes = ['coral']
    class_colours = {'coral': [0, 0, 255]}

    # Process only the first 100000 images with custom confidence threshold
    predictor = CoralSpawnPredictor(
        weights_path, 
        img_dir, 
        save_dir, 
        classes, 
        class_colours,
        iou_thresh=0.3,     # IOU threshold for NMS
        conf_thresh=0.25,   # Confidence threshold for detections
        max_images=200000   # Maximum number of images to process
    )
    predictor.run()