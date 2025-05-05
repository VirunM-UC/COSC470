from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 250

def load_images(image_folder, len_dataset):
    image_list = [Image.open(image_folder + f"image_{i}.jpg") for i in range(len_dataset)]
    return image_list

def read_excel(excel_fname):
    df = pd.read_excel(excel_fname) #df = pd.read_excel(excel_fname, keep_default_na = False) stop "NA" being auto converted to NaN
    df["City_Name"] = df["City_Name"].astype("string")

    df.loc[df["structure_type"].map(lambda x: isinstance(x, str)), "structure_type"] = df.loc[df["structure_type"].map(lambda x: isinstance(x, str)), "structure_type"].map(lambda s: s.lower())

    df[["sill_height", "structure_type", "building_conditions", "building_material", "occupancy", "roof_type", "landuse", "slope", "street_description"]] = \
        df[["sill_height", "structure_type", "building_conditions", "building_material", "occupancy", "roof_type", "landuse", "slope", "street_description"]].astype("category")
    
    return df

def clear_lost_images(df, image_mask_fname):
    image_mask = pd.read_csv(image_mask_fname)
    df = df.loc[image_mask.iloc[:,0]]
    return df

def main(image_folder, data_folder, excel_fname, image_mask_fname):
    df = read_excel(excel_fname)
    df["image"] = load_images(image_folder, len(df))
    df = clear_lost_images(df, image_mask_fname)
    not_test_df, test_df  = train_test_split(df, test_size = 0.2, random_state = RANDOM_STATE)
    train_df, val_df = train_test_split(not_test_df, test_size = 0.25, random_state = RANDOM_STATE)
    train_df.to_pickle(data_folder + "training.pkl")
    val_df.to_pickle(data_folder + "validation.pkl")
    test_df.to_pickle(data_folder + "testing.pkl")
        

if __name__ == '__main__':
    image_folder = 'images/'
    data_folder = "data/"
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask.csv"
    main(image_folder, data_folder, excel_fname, image_mask_fname)