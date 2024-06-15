from ultralytics import YOLO
from random import random
import time

model = YOLO("yolov8n.pt")

SIZES = [160, 320, 480, 640, 800, 960, 1120, 1280]

while True:
    try:
       model.train(
            data='data.yaml', epochs=20, batch=16,
            imgsz=SIZES[int(random()*7)],
            box= random() * 10,
            cls= random() * 10,
            dfl= random() * 10,
            label_smoothing = random(),
            nbs= 64,
            hsv_h= 0.015,
            hsv_s= random(),
            hsv_v= random(),
            degrees= -180 + random() * 360,
            translate= random(),
            scale= random() * 2,
            shear= -180 + random() * 360,
            perspective= random() * 0.001,
            flipud= random(),
            fliplr= random(),
            bgr= 0.0,
            mosaic= random(),
            mixup= random(),
            copy_paste= random(),
            erasing= random() * 0.9,
            crop_fraction= random() * 0.9 + 0.1
           )
    except:
        time.sleep(1)
"""
box: 4
cls: 1
dfl: 1.5
label_smoothing: 0.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0
scale: 0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0
bgr: 0.0
mosaic: 0
mixup: 0.0
copy_paste: 0.0
erasing: 0
crop_fraction: 1.0
"""