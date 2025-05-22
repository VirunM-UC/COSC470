from transformers import AutoImageProcessor
import datasets
from datasets import Dataset
from transformers import DefaultDataCollator
import evaluate

import numpy as np
import pandas as pd
import random
import math

#def data_map(feature_dict):
#   return (feature_dict["image"], feature_dict["building_material"])

#data to hf dataset
#labels = ["brick", "cinder", "steel", "under construction", "wood"]
labels = ["attached", "semi-detached", "detached"] 
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


def df_to_hfds_structure_type(df, mode):
    """
    Pandas dataframe to huggingface dataset for classifying structure_type.
    Args:
    df: dataframe
    mode: string, either "train" or "validate"
    """
    df = df[~df["structure_type"].isna()] #filter NaNs
    structure_types = df["structure_type"].map(lambda x: int(label2id[x])).astype("uint8")
    images = df.loc[:, "image"]
    df_data = pd.DataFrame({"image": images, "structure_type": structure_types})
    
    if mode == "train":
        print("train size (original): ", len(df_data))
    elif mode == "validate":
        print("validate size: ", len(df_data))
    print(df_data["structure_type"].value_counts())

    #Upsampling
    if mode == "train":
        df_classes = []
        for i in range(len(labels)):
            df_classes.append(df_data.loc[df_data["structure_type"] == i])
        max_index = max(range(len(labels)), key = lambda x: len(df_classes[x]))
        for i in range(len(labels)):
            if i == max_index:
                continue
            df_classes[i] = df_classes[i].sample(n = len(df_classes[max_index]), replace=True)
        df_data = pd.concat(df_classes)
        print("train_size (upsampled): ", len(df_data))
        print(df_data["structure_type"].value_counts())

    ds = Dataset.from_dict({"image": df_data["image"],
                             "label": df_data["structure_type"]}, 
                             
                             features = datasets.Features({"image": datasets.Image(),
                                                           "label": datasets.Value(dtype="uint8")}))
    return ds



def load_data(data_folder, file_name):
    """Takes a pickled dataframe and returns a Tensorflow Dataset
    """
    df = pd.read_pickle(data_folder + file_name)
    return df


df_train = load_data("data/", "training.pkl")
df_validate = load_data("data/", "validation.pkl")


hf_train = df_to_hfds_structure_type(df_train, mode = "train")
hf_validate = df_to_hfds_structure_type(df_validate, mode = "validate")

#preprocessor
checkpoint = "google/vit-base-patch16-224-in21k" #ViT (base: 350MB)
#checkpoint = "microsoft/swinv2-base-patch4-window16-256" #Swin Transformer V2 (base: 350MB)
#checkpoint = "facebook/convnext-base-224" #ConvNeXT (base: 350MB)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

from torchvision.transforms import Resize, Compose, Normalize, ToTensor

def preprocess_maker(processor):
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([Resize(size), ToTensor(), normalize])
    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    
    return transforms

hf_train.set_transform(preprocess_maker(image_processor))
hf_validate.set_transform(preprocess_maker(image_processor))
data_collator = DefaultDataCollator()

#metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_result = accuracy.compute(predictions=predictions, references=labels)

    f = f1.compute(predictions=predictions, references=labels, average=None)    
    f_result = dict()
    for index, value in enumerate(f["f1"]):
        f_result[f"f1_{index}"] = value
    
    result = acc_result | f_result
    return result



from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

#model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    ignore_mismatched_sizes = True,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


#hyperparameters
training_args = TrainingArguments(
    output_dir="ViT-structure_type",
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
    compute_metrics=compute_metrics,
)

trainer.train()