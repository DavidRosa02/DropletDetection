import matplotlib.pyplot as plt
import cv2
import json
import supervision as sv
from inference import get_model
import numpy as np
import pandas as pd

image = cv2.imread("img2.png")
model = get_model(model_id="droplet-detection-my1rz/9", api_key="OLUKLjYzlIrdzHYZTVZq")

results = model.infer(image, conf = 0.1, verbose = False, max_det = 1000)
global res
res = []
if len(results[0].predictions) == 300:
    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = model.infer(image_slice, conf = 0.1, verbose = True, max_det = 10000)
        size = len(results[0].predictions)
        for i in range(size):
            res.append(results[0].predictions[i].__dict__)
        return sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    
    slicer = sv.InferenceSlicer(callback = callback, slice_wh = [380, 380], overlap_ratio_wh = [0.32,0.32])
    detections = slicer(image)
else:
    for i in range(size):
        res.append(results[0].predictions[i].__dict__)
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
with open("data.json", "w") as f:
        json.dump(res, f)
data = json.load(open('data.json', 'r'))
df = pd.DataFrame(data)
df['area'] = df['width'] * df['height'] 
ax = df.plot.hist(column = ["area"], bins = 30)
print(len(data))
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
sv.plot_image(annotated_image)
