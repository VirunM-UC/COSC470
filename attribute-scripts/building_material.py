from transformers import AutoImageProcessor
import datasets
from datasets import Dataset
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw



#semantic preprocessing functions for use in df_to_hfds_building_material
def crop_fn(row):
    #row has index ["image", COLUMN_NAME, "xmin", "ymin", "xmax", "ymax", "score", "dist_squared"]
    row["image"] = row["image"].crop((row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
    return row 

def mask_fn(row):
    #row has index ["image", COLUMN_NAME, "xmin", "ymin", "xmax", "ymax", "score", "dist_squared"]
    im2 = Image.merge("RGB", [Image.effect_noise(size=row["image"].size, sigma= 5) for _ in range(3)])
    mask = Image.new("L", row["image"].size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([row["xmin"], row["ymin"], row["xmax"], row["ymax"]], fill=255)
    row["image"] = Image.composite(row["image"], im2, mask)
    return row 

def segmask_fn(row):
    #row has index ["image", COLUMN_NAME, "mask"]
    im2 = Image.merge("RGB", [Image.effect_noise(size=row["image"].size, sigma= 5) for _ in range(3)])
    row["image"] = Image.composite(row["image"], im2, row["mask"])
    return row

def df_to_hfds_building_material(df, mode, df_bounding_boxes = None, df_segmasks = None):
    """
    Pandas dataframe to huggingface dataset for classifying building material.
    Args:
    df: dataframe
    mode: string, either "train" or "validate"
    """
    #df = df[df["City_Name"].map(lambda x: (x in CITIES))] #filter everything that is not in CITIES

    df_data = df.loc[:, ["image", COLUMN_NAME]]
    df_data = df_data[df_data[COLUMN_NAME].map(lambda x: (x in LABELS))] #filter everything that is not in LABELS
    df_data[COLUMN_NAME] = df_data[COLUMN_NAME].map(lambda x: int(LABEL2ID[x])).astype("uint8")

    if df_bounding_boxes is not None:
        #cropping/masking bounding boxes
        df_data = df_data.join(df_bounding_boxes, how="inner")
        df_data = df_data[~pd.isna(df_data.iloc[:,2])] #filter out NA values (ones with no bounding box)
        df_data = df_data.apply(crop_fn, axis="columns", result_type="broadcast")
        df_data = df_data.loc[:, ["image", COLUMN_NAME]]
    elif df_segmasks is not None:
        df_data = df_data.join(df_segmasks, how="inner")
        df_data = df_data.apply(segmask_fn, axis="columns", result_type="broadcast")
        df_data = df_data.loc[:, ["image", COLUMN_NAME]]


    if mode == "train":
        print("train size (original): ", len(df_data))         
    elif mode == "validate":
        print("validate size: ", len(df_data))
    print(df_data[COLUMN_NAME].value_counts())

    #Upsampling
    if mode == "train":
        df_data = utils.upsample(df_data, LABELS, COLUMN_NAME)
        print("train_size (upsampled): ", len(df_data))
        print(df_data[COLUMN_NAME].value_counts())

    ds = Dataset.from_dict({"image": df_data["image"],
                            "label": df_data[COLUMN_NAME]}, 
                             
                            features = datasets.Features({"image": datasets.Image(),
                                                           "label": datasets.Value(dtype="uint8")}))
    return ds



from torchvision.transforms import Resize, Compose, Normalize, ToTensor

def preprocess_maker(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    _transforms = Compose([Resize(size), ToTensor(), normalize])
    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    
    return transforms


def main(model_name, data_folder, model_output_dir, bounding_boxes_fname = None, segmasks_fname = None):
    #data
    df_train = utils.load_data(data_folder, "training.pkl")
    df_validate = utils.load_data(data_folder, "validation.pkl")

    #semantic preprocessing
    if bounding_boxes_fname is not None:
        df_bounding_boxes = pd.read_csv(bounding_boxes_fname, index_col = 0)
        df_segmasks = None
    elif segmasks_fname is not None:
        df_bounding_boxes = None
        df_segmasks = pd.read_pickle(segmasks_fname)
    else:
        df_bounding_boxes = None
        df_segmasks = None

    hf_train = df_to_hfds_building_material(df_train, mode = "train", df_bounding_boxes = df_bounding_boxes, df_segmasks = df_segmasks)
    hf_validate = df_to_hfds_building_material(df_validate, mode = "validate")

    #preprocessor
    checkpoint = utils.CHECKPOINT[model_name]
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    hf_train.set_transform(preprocess_maker(image_processor))
    hf_validate.set_transform(preprocess_maker(image_processor))
    data_collator = DefaultDataCollator()

    #model
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes = True,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


    #hyperparameters
    training_args = TrainingArguments(
        output_dir= model_output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=hf_train,
        eval_dataset=hf_validate,
        processing_class=image_processor,
        compute_metrics=utils.default_metric_maker(ID2LABEL),
    )

    trainer.train()

if __name__ == "__main__":
    #LABELS = ["beton", "briques"] #LABELS = ["beton", "briques", "bois"] 
    #COLUMN_NAME = "mur" 
    LABELS = ["cinder", "brick"] 
    COLUMN_NAME = "building_material"
    #LABELS = ["attached", "semi-detached", "detached"] 
    #COLUMN_NAME = "structure_type"
    LABEL2ID, ID2LABEL = dict(), dict()
    for i, label in enumerate(LABELS):
        LABEL2ID[label] = str(i)
        ID2LABEL[str(i)] = label
    #CITIES = []
    
    model_name = "convnext"
    data_folder = "data-folders/material-data/"
    #bounding_boxes_fname = "bounding_boxes.csv"
    #segmasks_fname = "semantic_masks.pkl"
    model_output_dir = f"model-folders/{model_name}-building_material-model"
    main(model_name, data_folder, model_output_dir)