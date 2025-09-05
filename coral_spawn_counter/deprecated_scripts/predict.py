#! /usr/bin/env/python3

""" run model on folder of images
save predicted bounding box results both as txt and .jpg
"""

from ultralytics import YOLO
import os
import glob
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

weights_file_path = '/home/java/Java/ultralytics/runs/detect/train6/weights/best.pt'
img_folder = '/home/java/Java/data/cslics_desktop_data/20241119_cslics_desktop_aken/clean'
save_dir = '/home/java/Java/data/cslics_desktop_data/20241119_cslics_desktop_aken/detections'


classes = ["Four-Eight-Cell Stage", "First Cleavage", "Two-Cell Stage", "Advanced Stage", "Damaged", "Egg", "Larvae"]

orange = [0, 128, 225]
cyan = [225, 212, 0]
purple = [255, 0, 170]
yellow = [0, 255, 255]
brown = [2, 65, 144]
green = [0, 255, 0]
dark_purple = [112, 21, 97]

class_colours = {
    classes[0]: orange,
    classes[1]: cyan,
    classes[2]: purple,
    classes[3]: yellow,
    classes[4]: brown,
    classes[5]: green,
    classes[6]: dark_purple
}


def save_image_predictions_bb(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image as bounding box
    """
    img = cv.imread(imgname)
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
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 1) #cv2.rectangle(image, start_point, end_point, color, thickness)
        cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, class_colours[classes[cls]], 1) #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])  
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, img)
    return True

def save_txt_predictions_bb(predictions, imgname, txtsavedir):
    """
    save predictions/detections [xn1, yn1, xn2, yn2, cls, conf]) as bounding box (x and y values normalised)
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

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

# get predictions
print('Model Inference:')

imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
imgsave_dir = os.path.join(save_dir, 'detections_images')
txtsave_dir = os.path.join(save_dir, 'detections_txt')
os.makedirs(imgsave_dir, exist_ok=True)
os.makedirs(txtsave_dir, exist_ok=True)

for i, imgname in enumerate(imglist):
    print(f'predictions on {i+1}/{len(imglist)}')

    results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, max_det=1000)
    boxes = results[0].boxes 
    pred = []
    for b in boxes:
        if torch.cuda.is_available():
            xyxyn = b.xyxyn[0]
            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
    predictions = torch.tensor(pred)
    save_image_predictions_bb(predictions, imgname, imgsave_dir, class_colours, classes)
    save_txt_predictions_bb(predictions, imgname, txtsave_dir)

print('Done')


import code
code.interact(local=dict(globals(), **locals()))