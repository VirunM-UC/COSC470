from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd

from utils import make_url
from utils import KEY

RANDOM_STATE = 250

def get_image(point_x, point_y):
    resp = urlopen(make_url(location = f"{point_y},{point_x}", 
                       size = "400x400",
                       fov = str(90), 
                       return_error_code = "true",
                       source = "outdoor",
                       key = KEY))
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"image_{index}.jpg"), image)

def main(image_folder, csv_fname, included_indices_fname, sample = False):
    missing = []
    df = pd.read_csv(csv_fname)

    if sample == True:
        #collect included indices
        classes = []
        for i in ["beton", "briques", "bois"]:
            mask = (df["mur"] == i)
            class_indices = pd.Series([i for i in range(len(df)) if mask.iloc[i]])
            class_indices = class_indices.sample(n = 2500, replace = False, random_state = RANDOM_STATE)
            classes.append(class_indices)
        included_indices = pd.concat(classes)
    else:
        included_indices = pd.series(range(len(df)))
    print("Number of included records: ", len(included_indices))

    image_mask = pd.Series([True for _ in range(len(df))], dtype="boolean")
    for i in included_indices: #len(df)
        try:
            image = get_image(df.loc[i, "lon"], df.loc[i, "lat"])
            save_image(image, image_folder, i)
        except:
            image = np.zeros((400,400,3))
            save_image(image, image_folder, i)

            missing.append(i)
            image_mask.iloc[i] = False
    print(f"Missing: {len(missing)} out of {len(included_indices)} ({len(missing)/len(included_indices):.0%})")
    print(missing)
    
    included_indices = included_indices[included_indices.iloc[:, 0].map(lambda x: image_mask.iloc[x,0])]
    included_indices.to_csv(included_indices_fname, index = False)
        
        

if __name__ == '__main__':
    image_folder = 'image-folders/paris-images/'
    csv_fname = "paris_records.csv"
    included_indices_fname = "included_indices_paris.csv"
    main(image_folder, csv_fname, included_indices_fname, sample = False)

