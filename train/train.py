from ultralytics import YOLO


if __name__ != '__main__': exit()

resolutions = [
    #320,
    #400,
    #480,
    #560,
    #640,
    #720,
    #800,
    #880,
    #960,
    #1040,
    #1120,
    #1200,
    1280,
    #1360,
    1440
]
for resolution in resolutions:
    model = YOLO("yolov8n.pt")
    model.train(
        data='data.yaml', epochs=16, batch=24,
        imgsz=resolution,
        name=f"000_{resolution}",
        workers=0,
        hsv_h= 0.,
        hsv_s= 0,
        hsv_v= 0,
        degrees= 0,
        translate= 0,
        scale= 0,
        shear= 0.25,
        perspective= 0.001,
        flipud= 0.0,
        fliplr= 0.0,
        bgr= 0.0,
        mosaic= 0,
        mixup= 0,
        copy_paste= 0,
        erasing= 0.4,
        crop_fraction= 0.35
        )