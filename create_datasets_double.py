from PIL import Image
import numpy as np

import pandas as pd
pd.options.mode.copy_on_write = True

from sklearn.model_selection import train_test_split

RANDOM_STATE = 250

def load_images(image_folder, len_dataset):
    image_list = [Image.open(image_folder + f"image_{i}.jpg") for i in range(len_dataset)]
    return image_list

def read_excel(excel_fname):
    df = pd.read_excel(excel_fname) #df = pd.read_excel(excel_fname, keep_default_na = False) stop "NA" being auto converted to NaN
    
    table = str.maketrans("","","0123456789")
    df["City_Name"] = df["City_Name"].map(lambda s: s.translate(table).lower())
    df["City_Name"] = df["City_Name"].astype("category")

    df.loc[df["structure_type"].map(lambda x: isinstance(x, str)), "structure_type"] = df.loc[df["structure_type"].map(lambda x: isinstance(x, str)), "structure_type"].map(lambda s: s.lower())

    df[["structure_type", "building_conditions"]] = \
        df[["structure_type", "building_conditions"]].astype("category")
    
    return df

def clear_lost_images(df, image_mask_fname):
    image_mask = pd.read_csv(image_mask_fname)
    df = df.loc[image_mask.iloc[:,0]]
    return df

def main(image_folder, data_folder, excel_fname, image_mask_fname):
    df = read_excel(excel_fname)
    df = clear_lost_images(df, image_mask_fname)
    df["image"] = np.nan
    df["image"] = df["image"].astype("object")
    added_rows = []
    for i in df.index:
        copy = df.loc[i]
        df.loc[i, "image"] = Image.open(image_folder + f"image_{i}_0.jpg")
        try:
            copy["image"] = Image.open(image_folder + f"image_{i}_1.jpg")
        except FileNotFoundError:
            pass
        else:
            added_rows.append(copy)
    df = pd.concat([df, pd.DataFrame(added_rows)])
    
    #dataset splitting
    not_test_df, test_df  = train_test_split(df, test_size = 0.2, random_state = RANDOM_STATE, stratify = df["City_Name"])
    train_df, val_df = train_test_split(not_test_df, test_size = 0.25, random_state = RANDOM_STATE, stratify = not_test_df["City_Name"])
    train_df.to_pickle(data_folder + "training.pkl")
    val_df.to_pickle(data_folder + "validation.pkl")
    test_df.to_pickle(data_folder + "testing.pkl")
        

if __name__ == '__main__':
    #These 3 should change if using composite
    image_folder = 'image-folders/double-images/'
    data_folder = "data-folders/double-data/"
    image_mask_fname = "lost_images_mask_double.csv"

    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    main(image_folder, data_folder, excel_fname, image_mask_fname)