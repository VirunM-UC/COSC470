from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd

from utils import KEY

RANDOM_STATE = 250

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

def main(image_folder, csv_fname, image_mask_fname, included_indices_fname):
    missing = []
    df = pd.read_csv(csv_fname)

    #collect included indices
    classes = []
    for i in ["beton", "briques", "bois"]:
        mask = (df["mur"] == i)
        class_indices = pd.Series([i for i in range(len(df)) if mask.iloc[i]])
        class_indices = class_indices.sample(n = 2500, replace = False, random_state = RANDOM_STATE)
        classes.append(class_indices)
    included_indices = pd.concat(classes)
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
    image_mask.to_csv(image_mask_fname, index = False)
    included_indices.to_csv(included_indices_fname, index = False)
        
        

if __name__ == '__main__':
    image_folder = 'image-folders/french-images/'
    csv_fname = "french_records.csv"
    image_mask_fname = "lost_images_mask_french.csv"
    included_indices_fname = "included_indices.csv"
    main(image_folder, csv_fname, image_mask_fname, included_indices_fname)

