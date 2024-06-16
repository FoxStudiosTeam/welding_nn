from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

BATCH_SIZE = 32
PATH = "weights.pt"
model = YOLO(PATH)
if __name__ != '__main__': exit()
dirs = ["datasets/evaluate", "datasets/train"]
files = []
for dir in dirs:
    files += [f"{dir}/{f}" for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith("jpg") ]

filenames = []
ids = []
rel_x = []
rel_y = []
width = []
height = []

batch = []
for file in files:
    batch.append(file)
    if len(batch) == BATCH_SIZE or file == files[-1]:
        results = model(batch)
        i = 0
        for result in results:
            for box in result.boxes:
                filenames.append(batch[i].split("/")[-1])
                ids.append(int(box.cls.cpu().numpy()[0]))
                x, y, w, h = box.xyxyn.cpu().numpy()[0]
                rel_x.append(x)
                rel_y.append(y)
                width.append(w)
                height.append(h)
            i+=1
        batch = []


df = pd.DataFrame({
    "filename": filenames,
    "class_id": ids,
    "rel_x": rel_x,
    "rel_y": rel_y,
    "width": width,
    "height": height
})


df.to_csv(f"./sumbission{PATH.split("/")[4]}.csv", index=False, sep=";")