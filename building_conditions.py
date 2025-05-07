from transformers import AutoImageProcessor
import datasets
from datasets import Dataset
from transformers import DefaultDataCollator
import evaluate

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import math
from tensorflow import keras

#def data_map(feature_dict):
#   return (feature_dict["image"], feature_dict["building_material"])

#data to hf dataset
#labels = ["brick", "cinder", "steel", "tile", "under construction", "wood"]
labels = ["very poor", "poor", "fair", "good", "very good"] 
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


def df_to_hfds_building_conditions(df, mode):
    """
    Pandas dataframe to huggingface dataset for classifying building_conditions.
    Args:
    df: dataframe
    mode: string, either "train" or "validate"
    """
    df = df[~df["building_conditions"].isna()] #filter NaNs
    building_conditions = df["building_conditions"].apply(lambda x: int(label2id[x])).astype("uint8")
    images = df.loc[:, "image"]
    df_data = pd.DataFrame({"image": images, "building_conditions": building_conditions})
    
    if mode == "train":
        print("train size (original): ", len(df_data))
    elif mode == "validate":
        print("validate size: ", len(df_data))
    print(df_data["building_conditions"].value_counts())

    if mode == "train":
        df_classes = []
        for i in range(len(labels)):
            df_classes.append(df_data.loc[df_data["building_conditions"] == i])
        max_index = max(range(len(labels)), key = lambda x: len(df_classes[x]))
        for i in range(len(labels)):
            if i == max_index:
                continue
            df_classes[i] = df_classes[i].sample(n = len(df_classes[max_index]), replace=True)
        df_data = pd.concat(df_classes)
        print("train_size (upsampled): ", len(df_data))
        print(df_data["building_conditions"].value_counts())


    ds = Dataset.from_dict({"image": df_data["image"],
                             "building_conditions": df_data["building_conditions"]}, 
                             
                             features = datasets.Features({"image": datasets.Image(),
                                                           "building_conditions": datasets.Value(dtype="uint8")}))
    return ds



def load_data(data_folder, file_name):
    """Takes a pickled dataframe and returns a Tensorflow Dataset
    """
    df = pd.read_pickle(data_folder + file_name)
    return df


df_train = load_data("data/", "training.pkl")
df_validate = load_data("data/", "validation.pkl")


hf_train = df_to_hfds_building_conditions(df_train, mode = "train")
hf_validate = df_to_hfds_building_conditions(df_validate, mode = "validate")
data_collator = DefaultDataCollator(return_tensors="tf")

#preprocessor
checkpoint = "google/vit-base-patch16-224-in21k" #ViT
#checkpoint = "microsoft/swinv2-base-patch4-window16-256" #Swin Transformer V2 (not implemented in TensorFlow)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

def preprocess_maker(processor):
    size = (processor.size["height"], processor.size["width"])
    def preprocess(example_batch):
        example_batch["image"] = [np.array(image) for image in example_batch["image"]]
        
        example_batch["image"] = tf.image.resize(example_batch["image"], size)
        example_batch["image"] = [tf.transpose(image) for image in example_batch["image"]]
        return example_batch
    return preprocess

hf_train.set_transform(preprocess_maker(image_processor))
hf_validate.set_transform(preprocess_maker(image_processor))

#metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_result = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels, average=None, zero_division=0)
    prec_result = dict()
    for index, value in enumerate(prec["precision"]):
        prec_result[f"precision_{index}"] = value
    rec_result = recall.compute(predictions=predictions, references=labels, average=None)
    result = acc_result | (prec_result | rec_result)
    return result



#hyperparameters
from transformers import create_optimizer
batch_size = 16
num_epochs = 5
num_train_steps = math.ceil(len(hf_train) / batch_size) * num_epochs #?
learning_rate = 3e-5
weight_decay_rate = 0.01
optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)


#model
from transformers import TFAutoModelForImageClassification
model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

#hf to tf dataset
tf_train_dataset = hf_train.to_tf_dataset(
    columns="image", label_cols="building_conditions", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
tf_eval_dataset = hf_validate.to_tf_dataset(
    columns="image", label_cols="building_conditions", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

#loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy
loss = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

#callbacks
from transformers.keras_callbacks import KerasMetricCallback
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
callbacks = [metric_callback]


model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)