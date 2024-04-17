import matplotlib.pyplot as plt
import cv2
import json
import supervision as sv
from inference import get_model
import numpy as np
import pandas as pd
from PIL import Image
import glob

model = get_model(model_id="droplet-detection-my1rz/9", api_key="OLUKLjYzlIrdzHYZTVZq")
files = glob.glob('images/*.png')
file_name = []
for i in range(len(files)):
    file_name.append(files[i][7:])
for j in range(len(files)):
    image = cv2.imread(files[j])
    img = Image.open(files[j])
    width, height = img.size
    global res
    res = []
    if width * height > 300000:
        def callback(image_slice: np.ndarray) -> sv.Detections:
            results = model.infer(image_slice, conf = 0.01, verbose = True, max_det = 10000)
            size = len(results[0].predictions)
            for i in range(size):
                res.append(results[0].predictions[i].__dict__)
            return sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    
        slicer = sv.InferenceSlicer(callback = callback, slice_wh = [380, 380], overlap_ratio_wh = [0.32,0.32])
        detections = slicer(image)
    else:
        results = model.infer(image, conf = 0.01, verbose = True, max_det = 10000)
        size = len(results[0].predictions)
        for i in range(size):
            res.append(results[0].predictions[i].__dict__)
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    with open(f'data/{file_name[j]}_data{j}.json', "w") as f:
        json.dump(res, f)
    data = json.load(open(f'data/{file_name[j]}_data{j}.json', 'r'))
    df = pd.DataFrame(data)
    df['area'] = df['width'] * df['height'] 
    ax = df.plot.hist(column = ["area"], bins = 30)
    fig = ax.get_figure()
    fig.savefig(f'Size_Plots/{file_name[j]}_figure{j}.png')

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    sv.plot_image(annotated_image)
    with sv.ImageSink(target_dir_path='Annotated_Images', overwrite = False, image_name_pattern=f'{file_name[j]}_annotated') as sink:
        sink.save_image(image=annotated_image)