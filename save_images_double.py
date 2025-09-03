from urllib.request import urlopen
from urllib.error import URLError
import cv2
import numpy as np
import pandas as pd
import json

from utils import make_url
from utils import KEY, SESSION

def get_double_images(point_x, point_y):
    #get first image
    resp = urlopen(make_url(location = f"{point_y},{point_x}", 
                       size = "400x400",
                       fov = str(120), 
                       return_error_code = "true",
                       source = "outdoor",
                       key = KEY))
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    #get panorama metadata
    metadata_url = make_url(api = "tile", metadata = True, 
                            session = SESSION, 
                            key = KEY, 
                            lat = str(point_y), 
                            lng = str(point_x),
                            radius = str(50))
    resp = urlopen(metadata_url)
    metadata_json = json.loads(resp.read())

    panoramas = metadata_json["links"]
    #collect distances of linked panoramas from building
    build_location = {"lat": float(point_y), "lng": float(point_x)}
    for pan in panoramas:
        resp = urlopen(make_url(api = "static", metadata = True, pano = pan["panoId"], key = KEY))
        pan_json = json.loads(resp.read())
        pan["location"] = pan_json["location"]
        pan["distance"] = utils.distance(build_location, pan_json["location"])

    #decision to include extra image or not
    if len(panoramas) > 0:
        closest = min(panoramas, key = lambda x: x["distance"])
        if closest["distance"] < 50:
            heading = utils.heading(build_location, closest["location"])
            try:
                resp = urlopen(make_url(pano = closest["panoId"], 
                       size = "400x400",
                       fov = str(120), 
                       return_error_code = "true",
                       source = "outdoor",
                       heading = str(heading),
                       key = KEY))
            except URLError:
                pass
            else:
                image2 = np.asarray(bytearray(resp.read()), dtype='uint8')
                image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
                return (image, image2)

    return (image,)


def save_image(image, image_folder, index, pan_index):
    cv2.imwrite((image_folder + f"image_{index}_{pan_index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    missing = []
    df = pd.read_excel(excel_fname)
    data_mask = pd.Series([True for _ in range(len(df))], dtype="boolean")
    for i in range(len(df)):
        try:
            images = get_double_images(df.loc[i, "POINT_X"], df.loc[i, "POINT_Y"])
            for j in range(len(images)):
                save_image(images[j], image_folder, i, j)
        except:
            image = np.zeros((400,400,3))
            save_image(image, image_folder, i, 0)

            missing.append(i)
            data_mask.iloc[i] = False
    print(f"Missing: {len(missing)} out of {len(df)} ({len(missing)/len(df):.0%})")
    print(missing)
    data_mask.to_csv(image_mask_fname, index = False)

   

if __name__ == '__main__':
    image_folder = 'image-folders/double-images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask_double.csv"
    main(image_folder, excel_fname, image_mask_fname)
