from urllib.request import urlopen
from urllib.error import URLError
import cv2
import numpy as np
import pandas as pd
import json

from utils import make_url
from utils import KEY, SESSION

def get_pano(point_x, point_y):
    metadata_url = make_url(api = "tile", metadata = True, 
                            session = SESSION, 
                            key = KEY, 
                            lat = str(point_y), 
                            lng = str(point_x),
                            radius = str(50))
    resp = urlopen(metadata_url)
    metadata_json = json.loads(resp.read())
    panoId = metadata_json["panoId"]

    resp = urlopen(make_url(
        api = "tile", metadata = False,
        session = SESSION, 
        key = KEY,
        panoId = panoId
    ))
    pano_image = np.asarray(bytearray(resp.read()), dtype='uint8')
    pano_image = cv2.imdecode(pano_image, cv2.IMREAD_COLOR)
    return pano_image

def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"pano_{index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    missing = []
    df = pd.read_excel(excel_fname)
    #data_mask = pd.Series([True for _ in range(len(df))], dtype="boolean")
    for i in range(10): #len(df)
        try:
            image = get_pano(df.loc[i, "POINT_X"], df.loc[i, "POINT_Y"])
            save_image(image, image_folder, i)
        except URLError:
            image = np.zeros((400,400,3))
            save_image(image, image_folder, i)

            #missing.append(i)
            #data_mask.iloc[i] = False
    #print(f"Missing: {len(missing)} out of {len(df)} ({len(missing)/len(df):.0%})")
    #print(missing)
    #data_mask.to_csv(image_mask_fname, index = False)
        
        

if __name__ == '__main__':
    image_folder = 'image-folders/test-images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask.csv"
    main(image_folder, excel_fname, image_mask_fname )