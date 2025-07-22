import pandas as pd
import numpy as np
from io import BytesIO

import evaluate

KEY = "AIzaSyDE50N-WPn4s06OKhccYdDPXVnJ_k6O0bM"

SESSION = "AJVsH2zr4yJoYwmzPr4ZwqEXJkYim51OtpmbIERTWXYHIWZ2W0mqPAwp6cHhCf9c-SKZZm2i1DfsAofbQCpCcnU6wA"

CHECKPOINT = {
    "vit": "google/vit-base-patch16-224-in21k", #ViT (base: 350MB)
    "swinv2": "microsoft/swinv2-base-patch4-window16-256", #Swin Transformer V2 (base: 350MB)
    "convnext": "facebook/convnext-base-224", #ConvNeXT (base: 350MB)
}

#Google Street View

def heading(build_loc, pan_loc):
    """
    Calculates the heading of the panorama that points to the building.

    Parameters:
    build_loc: building location represented as a dictionary of the form {"lat": float(latitude), "lng": float(longitude)}
    pano_loc: panorama location, represented the same way as build_loc
    """
    lat_diff = build_loc["lat"] - pan_loc["lat"]
    lng_diff = build_loc["lng"] - pan_loc["lng"]
    heading = np.arctan2(lng_diff, lat_diff) * 180/np.pi

    return heading


def distance(loc1, loc2):
    """
    Calculates the distance in metres between two points defined by lat/lng coordinates, using the haversine fromula.
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters:
    loc1, loc2: the two locations represented as dictionaries of the form {"lat": float(latitude), "lng": float(longitude)}
    """
    R = 6371e3 #metres

    #radian values
    phi1 = loc1["lat"] * np.pi/180
    phi2 = loc2["lat"] * np.pi/180
    delta_phi = (loc2["lat"] - loc1["lat"]) * np.pi/180
    delta_lambda = (loc2["lng"] - loc1["lng"]) * np.pi/180

    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*(np.sin(delta_lambda/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c

    return d


#Data

def load_data(data_folder, file_name):
    """
    Takes a pickled dataframe and returns a Pandas DataFrame
    """
    df = pd.read_pickle(data_folder + file_name)
    return df

def df_to_excel(df, file_path):
    """
    Saves a DataFrame with images into an excel file.
    """
    images = df.loc[:, "image"]
    df = df.iloc[:, :-1]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Get the xlsxwriter workbook and worksheet objects.
    worksheet = writer.sheets['Sheet1']

    # Insert an image.
    for i in range(len(df)):
        image = images.iloc[i]
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        worksheet.insert_image(f'P{i+2}', "", {'image_data': image_buffer})

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()


#Huggingface
def default_metric_maker(id2label):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc_result = accuracy.compute(predictions=predictions, references=labels)

        f = f1.compute(predictions=predictions, references=labels, average=None)    
        f_result = dict()
        f_result["f1_macro"] = sum(f["f1"]) / len(f["f1"])
        for index, value in enumerate(f["f1"]):
            f_result[f"f1_{id2label[str(index)]}"] = value
        
        result = acc_result | f_result
        return result

    return compute_metrics