from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd
import json

KEY = "AIzaSyDE50N-WPn4s06OKhccYdDPXVnJ_k6O0bM"


def make_url(metadata = False, **kwargs):
    """
    Creates a url request for Google Maps Static Street View API
    For example, https://maps.googleapis.com/maps/api/streetview/metadata?location={point_y},{point_x}&source=outdoor&key={KEY}

    Paramaters:
    metadata: Boolean for whether to return the image request or the metadata request.
    **kwargs: All the parameters to pass to the API with their values in string form.
    """
    base = "https://maps.googleapis.com/maps/api/streetview"
    if metadata == True:
        base += "/metadata"
    kwarg_strings = []
    for kwarg in kwargs:
        kwarg_str = kwarg + "=" +  kwargs[kwarg]
        kwarg_strings.append(kwarg_str)
    url = base + "?" + "&".join(kwarg_strings)
    return url

def get_image(point_x, point_y):
    #get panorama metadata
    metadata_url = make_url(metadata = True, location = f"{point_y},{point_x}", source = "outdoor", key = KEY)
    resp = urlopen(metadata_url)
    metadata_json = json.loads(resp.read())

    #calculate heading
    pan_location = metadata_json["location"]
    build_location = {"lat": float(point_y), "lng": float(point_x)}
    lat_diff = build_location["lat"] - pan_location["lat"]
    lng_diff = build_location["lng"] - pan_location["lng"]
    heading = np.arctan2(lng_diff, lat_diff) * 180/np.pi


    heading_diffs = [-90, 0, 90]
    images = []
    for diff in heading_diffs:
        url = make_url(location = f"{point_y},{point_x}", 
                       size = "400x100",
                       fov = str(120), 
                       return_error_code = "true",
                       source = "outdoor",
                       heading = str(heading + diff),
                       key = KEY)
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        images.append(image)
    
    comp_image = np.concatenate(images)
    return comp_image

def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"image_{index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    point_x = "-78.49973065"
    point_y = "-0.080093856"
    image = get_image(point_x, point_y)
    save_image(image, image_folder, "comp2")

        

if __name__ == '__main__':
    image_folder = 'test-images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask.csv"
    main(image_folder, excel_fname, image_mask_fname )
