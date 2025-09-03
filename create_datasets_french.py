from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 250

def main(image_folder, data_folder, data_fname, included_indices_fname):
    df = pd.read_csv(data_fname)
    included_indices = pd.read_csv(included_indices_fname)
    included_mask = pd.Series([False for _ in range(len(df))], dtype="boolean")
    included_mask.iloc[included_indices.iloc[:,0]] = True
    df = df.loc[included_mask]
    df["image"] = [Image.open(image_folder + f"image_{i}.jpg") for i in df.index]

    not_test_df, test_df  = train_test_split(df, test_size = 0.2, random_state = RANDOM_STATE, stratify = df["mur"])
    train_df, val_df = train_test_split(not_test_df, test_size = 0.25, random_state = RANDOM_STATE, stratify = not_test_df["mur"])
    train_df.to_pickle(data_folder + "training.pkl")
    val_df.to_pickle(data_folder + "validation.pkl")
    test_df.to_pickle(data_folder + "testing.pkl")
        

if __name__ == '__main__':
    #These 3 should change between paris or french
    image_folder = 'image-folders/paris-images/'
    data_folder = "data-folders/paris-data/"
    included_indices_fname = "included_indices_paris.csv"
    data_fname = "paris_records.csv"
    main(image_folder, data_folder, data_fname, included_indices_fname)