from urllib.request import urlopen
from urllib.error import URLError
import cv2
import numpy as np
import pandas as pd

from utils import make_url
from utils import KEY

def get_image(point_x, point_y):
    url = make_url(location = f"{point_y},{point_x}", 
                size = "400x400",
                fov = str(90), 
                return_error_code = "true",
                source = "outdoor",
                key = KEY)
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"image_{index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    missing = []
    df = pd.read_excel(excel_fname)
    data_mask = pd.Series([True for _ in range(len(df))], dtype="boolean")
    for i in range(len(df)): #len(df)
        try:
            image = get_image(df.loc[i, "POINT_X"], df.loc[i, "POINT_Y"])
            save_image(image, image_folder, i)
        except URLError:
            image = np.zeros((400,400,3))
            save_image(image, image_folder, i)

            missing.append(i)
            data_mask.iloc[i] = False
    print(f"Missing: {len(missing)} out of {len(df)} ({len(missing)/len(df):.0%})")
    print(missing)
    data_mask.to_csv(image_mask_fname, index = False)
        
        

if __name__ == '__main__':
    image_folder = 'image-folders/fov90-images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask_fov90.csv"
    main(image_folder, excel_fname, image_mask_fname )

