from transformers import pipeline
import datasets
from datasets import Dataset
import numpy as np
import pandas as pd
from PIL import Image


def select_box(predictions):
    if predictions == []:
        return {"xmin": None, "ymin": None, "xmax": None, "ymax": None, "score": None, "dist_squared": None}
    #find distance from midpoint
    x, y = (200, 200)
    for prediction in predictions:
        xmin, ymin, xmax, ymax = prediction["box"].values()
        closest_x = min(max(x, xmin), xmax)
        closest_y = min(max(y, ymin), ymax)
        dist_squared = (x - closest_x)**2 + (y - closest_y)**2
        prediction["dist_squared"] = dist_squared
    #select box
    min_dist = min([prediction["dist_squared"] for prediction in predictions])
    close_predictions = [prediction for prediction in predictions if prediction["dist_squared"] == min_dist] #exactly equal is questionable
    max_score_prediction = max(close_predictions, key = lambda d: d["score"])
    #additional info
    max_score_prediction["box"]["score"] = max_score_prediction["score"]
    max_score_prediction["box"]["dist_squared"] = max_score_prediction["dist_squared"]
    
    return max_score_prediction["box"]


def main(image_folder, bounding_boxes_fname):
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
    indices = list(range(1484)) #change for french #1484
    images = [{"image": Image.open(image_folder + f"image_{i}.jpg"), "candidate_labels": ["building"]} for i in indices]
    #ds = Dataset.from_dict({"image": images}, features = datasets.Features({"image": datasets.Image()}))
    
    predictions_list = detector(images)

    target_boxes = map(select_box, predictions_list)
    df = pd.DataFrame(target_boxes, index=indices)
    df.to_csv(bounding_boxes_fname)


if __name__ == '__main__':
    image_folder = 'image-folders/images/'
    bounding_boxes_fname = "bounding_boxes.csv"
    main(image_folder, bounding_boxes_fname)