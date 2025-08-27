from transformers import pipeline
import datasets
from datasets import Dataset
import numpy as np
import pandas as pd
from PIL import Image


def get_mask(predictions):
    predictions = [prediction["mask"] for prediction in predictions if prediction["label"] == "building"]
    if predictions == []:
        return None
    else:
        return predictions[0] 

def main(image_folder, semantic_masks_fname):
    checkpoint = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    detector = pipeline(model=checkpoint, task="image-segmentation")
    indices = list(range(1484)) #change for french #1484
    images = [Image.open(image_folder + f"image_{i}.jpg") for i in indices]
    
    predictions_list = detector(images)

    image_masks = map(get_mask, predictions_list)
    df = pd.DataFrame(image_masks, index=indices)
    df = df[~pd.isna(df.iloc[:,0])] #filter out NA values (ones with no building mask)
    df.to_pickle(semantic_masks_fname)

if __name__ == '__main__':
    image_folder = 'image-folders/images/'
    
    semantic_masks_fname = "semantic_masks.pkl"
    main(image_folder, semantic_masks_fname)
