from transformers import AutoImageProcessor
import datasets
from datasets import Dataset
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

import utils

import numpy as np
import pandas as pd
import random
import math

LABELS = ["very poor", "poor", "fair", "good", "very good"] 
LABEL2ID, ID2LABEL = dict(), dict()
for i, label in enumerate(LABELS):
    LABEL2ID[label] = str(i)
    ID2LABEL[str(i)] = label


def df_to_hfds_building_conditions(df, mode):
    """
    Pandas dataframe to huggingface dataset for classifying building_conditions.
    Args:
    df: dataframe
    mode: string, either "train" or "validate"
    """
    df = df[~df["building_conditions"].isna()] #filter NaNs
    building_conditions = df["building_conditions"].map(lambda x: int(LABEL2ID[x])).astype("float32")
    images = df.loc[:, "image"]
    df_data = pd.DataFrame({"image": images, "building_conditions": building_conditions})
    
    if mode == "train":
        print("train size (original): ", len(df_data))
    elif mode == "validate":
        print("validate size: ", len(df_data))
    print(df_data["building_conditions"].value_counts())

    if mode == "train":
        df_classes = []
        for i in range(len(LABELS)):
            df_classes.append(df_data.loc[df_data["building_conditions"] == i])
        max_index = max(range(len(LABELS)), key = lambda x: len(df_classes[x]))
        for i in range(len(LABELS)):
            if i == max_index:
                continue
            df_classes[i] = df_classes[i].sample(n = len(df_classes[max_index]), replace=True)
        df_data = pd.concat(df_classes)
        print("train_size (upsampled): ", len(df_data))
        print(df_data["building_conditions"].value_counts())


    ds = Dataset.from_dict({"image": df_data["image"],
                             "label": df_data["building_conditions"]}, 
                             
                             features = datasets.Features({"image": datasets.Image(),
                                                           "label": datasets.Value(dtype="float32")}))
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

#metrics
def output_to_int(x):
    """
    Takes model output predictions and turns it into a label ID
    """
    y = 0
    if x >= 2.5:
        y = 4
    elif x < 2.5 and x >= 2.0 :
        y = 3
    elif x < 2.0 and x >= 1.5:
        y = 2
    elif x < 1.5 and x >= 1.0:
        y = 1
    elif x < 1.0 :
        y= 0
    return y

def metric_maker():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        #both predictions and labels need to be turned to ints
        predictions, labels = eval_pred
        labels = labels.astype(np.uint8)
        predictions = np.array([output_to_int(x.item()) for x in predictions])
        acc_result = accuracy.compute(predictions=predictions, references=labels)

        f_result = f1.compute(predictions=predictions, references=labels, average="macro")    
        
        result = acc_result | f_result
        return result

    return compute_metrics


def main(model_name, data_folder):
    #data
    df_train = utils.load_data(data_folder, "training.pkl")
    df_validate = utils.load_data(data_folder, "validation.pkl")

    hf_train = df_to_hfds_building_conditions(df_train, mode = "train")
    hf_validate = df_to_hfds_building_conditions(df_validate, mode = "validate")

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
        num_labels=1,
    )


    #hyperparameters
    training_args = TrainingArguments(
        output_dir= f"{model_name}-building_conditions-comp_model",
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
        compute_metrics=metric_maker(),
    )

    trainer.train()

if __name__ == "__main__":
    model_name = "vit"
    data_folder = "data-folders/composite-data/"
    main(model_name, data_folder)