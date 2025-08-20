from urllib.request import urlopen
from urllib.error import URLError
import cv2
import numpy as np
import pandas as pd
import json

import utils
from utils import KEY


def make_url(api = "maps", metadata = False, **kwargs):
    """
    Creates a url request for two Google Maps APIs: the Static Street View API and the Map Tiles API.
    This is mainly  for formatting, and the actual syntax for the API request is the responsibility of the user, but is described below.

    Static Street View API
    Image request syntax is described here: https://developers.google.com/maps/documentation/streetview/request-streetview
    - EXAMPLE: "https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&return_error_code=true&source=outdoor&location={point_y},{point_x}&key={KEY}"
    Metadata request syntax is described here: https://developers.google.com/maps/documentation/streetview/metadata
    - EXAMPLE: "https://maps.googleapis.com/maps/api/streetview/metadata?location={point_y},{point_x}&source=outdoor&key={KEY}"

    Map Tiles API
    Metadata request is described here: https://developers.google.com/maps/documentation/tile/streetview?hl=en#street_view_metadata 
    - EXAMPLE: "https://tile.googleapis.com/v1/streetview/metadata?session={SESSION}&key={KEY}&lat={lat}&lng={lng}&radius={radius}"

    Paramaters:
    api: string, either "maps" or "tile", which selects the API to use.
    metadata: Boolean for whether to return the image request or the metadata request.
    **kwargs: All the parameters to pass to the API with their values in string form.
    
    Returns: the url request as a string.
    """
    if api == "maps":
        base = "https://maps.googleapis.com/maps/api/streetview"
        if metadata == True:
            base += "/metadata"
    elif api == "tile":
        base = "https://tile.googleapis.com/v1/streetview"
        if metadata == True:
            base += "/metadata"
        else:
            base += "/tiles/0/0/0"
    kwarg_strings = []
    for kwarg in kwargs:
        kwarg_str = kwarg + "=" +  kwargs[kwarg]
        kwarg_strings.append(kwarg_str)
    url = base + "?" + "&".join(kwarg_strings)
    return url

def get_pano(point_x, point_y):
    metadata_url = make_url(api = "tile", metadata = True, 
                            session = utils.SESSION, 
                            key = KEY, 
                            lat = str(point_y), 
                            lng = str(point_x),
                            radius = str(50))
    resp = urlopen(metadata_url)
    metadata_json = json.loads(resp.read())
    panoId = metadata_json["panoId"]

    resp = urlopen(make_url(
        api = "tile", metadata = False,
        session = utils.SESSION, 
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