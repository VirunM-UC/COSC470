from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd
import json

from utils import KEY


def make_url(api = "maps", metadata = False, **kwargs):
    """
    Creates a url request for Google Maps Static Street View API
    For example, https://maps.googleapis.com/maps/api/streetview/metadata?location={point_y},{point_x}&source=outdoor&key={KEY}
    or https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&return_error_code=true&source=outdoor&location={point_y},{point_x}&key={KEY}

    Paramaters:
    metadata: Boolean for whether to return the image request or the metadata request.
    **kwargs: All the parameters to pass to the API with their values in string form.
    """
    if api == "maps":
        base = "https://maps.googleapis.com/maps/api/streetview"
    elif api == "tile":
        base = "https://tile.googleapis.com/v1/streetview"
    else:
        base = "https://tile.googleapis.com/v1/streetview"

    if metadata == True:
        base += "/metadata"
    kwarg_strings = []
    for kwarg in kwargs:
        kwarg_str = kwarg + "=" +  kwargs[kwarg]
        kwarg_strings.append(kwarg_str)
    url = base + "?" + "&".join(kwarg_strings)
    return url

def get_double_image(point_x, point_y):
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
                            session = utils.SESSION, 
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
        resp = urlopen(make_url(api = "maps", metadata = True, pano = pan["panoId"], key = KEY))
        pan_json = json.loads(resp.read())
        pan["location"] = pan_json["location"]
        pan["distance"] = utils.distance(build_location, pan_json["location"])

    #decision to include extra image or not
    if len(panoramas) > 0:
        closest = min(panoramas, key = lambda x: x["distance"])
        if closest["distance"] < 50:
            heading = utils.heading(build_location, closest["location"])
            resp = urlopen(make_url(pano = closest["panoId"], 
                       size = "400x400",
                       fov = str(120), 
                       return_error_code = "true",
                       source = "outdoor",
                       heading = str(heading),
                       key = KEY))
            image2 = np.asarray(bytearray(resp.read()), dtype='uint8')
            image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
            return (image, image2)

    return (image,)


def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"image_{index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    """
    point_x = "-78.49973065"
    point_y = "-0.080093856"
    image = get_comp_image(point_x, point_y)
    save_image(image, image_folder, "45")

    """    
    pass
   

if __name__ == '__main__':
    image_folder = 'image-folders/composite-images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask_composite.csv"
    main(image_folder, excel_fname, image_mask_fname )
