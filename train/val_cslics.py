from ultralytics import YOLO
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results
 
# Load a model
model = YOLO('/home/dtsai/Data/cslics_desktop_datasets/models/cslics_desktop_embryogenesis_20251211_yolov8m_1280p/weights/cslics_desktop_embryogenesis_20251211_yolov8m_1280p.pt')  # load a custom model

# model = YOLO('');

# Validate the model
# metrics = model.val(data='cslics_desktop_embryogenesis_2023.yaml',
#             imgsz=1280,)  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category


# inference on a given folder:



img_dir = '/home/dtsai/Data/cslics_desktop_datasets/unprocessed_data/test'
# img_dir = '/home/dtsai/Data/cslics_desktop_datasets/cslics_desktop_October_2024_aken/split/images/val'
# img_dir = '/home/dtsai/Data/cslics_desktop_datasets/2023_combined_embryogenesis/split/images/test'
# img_dir = '/home/dtsai/Data/cslics_datasets/cslics_2025_nov/split/test/images'
img_list = sorted(glob.glob(os.path.join(img_dir,'*.jpg')))
print(f'img_list length = {len(img_list)}')

# out_dir = '/home/dtsai/Data/cslics_desktop_datasets/2023_combined_embryogenesis/split/images/test_detections'
# out_dir = '/home/dtsai/Data/cslics_desktop_datasets/cslics_desktop_October_2024_aken/split/images/val_detections'
out_dir = '/home/dtsai/Data/cslics_desktop_datasets/unprocessed_data/test_detections'
os.makedirs(out_dir, exist_ok=True)

max_det = 9999
conf_thresh=0.1
for i, img_name in enumerate(img_list):
    print(f'{i}/{len(img_list)}: {os.path.basename(img_name)}')
    results = model.predict(img_name, 
                    save=True, 
                    save_txt=True,
                    save_conf=True,
                    boxes=True,
                    conf=conf_thresh,
                    iou=0.5,
                    agnostic_nms=True,
                    max_det=max_det)
    # print(type(results))
 
    res: Results = results[0]
    res_plotted = res.plot(conf=conf_thresh,
                           font_size=20,
                           line_width=2,
                           labels=True,
                           boxes=True,
                           probs=True)
        # how to adjust the plotting characteristics
    res_rgb = cv.cvtColor(res_plotted, cv.COLOR_BGR2RGB)
    img_save_name = os.path.basename(img_name).rsplit('.')[0] + '_det.jpg'
    plt.imsave(os.path.join(out_dir, img_save_name), res_rgb)
    # plt.show()

    # TODO plot/annotate the total number of counts/class in the corner of the image
    

print("Done")
import code
code.interact(local=dict(globals(), **locals()))   

