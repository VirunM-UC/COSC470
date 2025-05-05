from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd

KEY = "AIzaSyDE50N-WPn4s06OKhccYdDPXVnJ_k6O0bM"
#MISSING = [107, 113, 363, 471, 480, 499, 588, 593, 597, 969, 1411, 1422, 1423, 1426, 1434, 1435, 1438, 1440, 1442, 1444, 1464, 1472, 1475, 1476]

def get_image(point_x, point_y):
    url = f"https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&return_error_code=true&location={point_y},{point_x}&key={KEY}"
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def save_image(image, image_folder, index):
    cv2.imwrite((image_folder + f"image_{index}.jpg"), image)

def main(image_folder, excel_fname, image_mask_fname ):
    missing = []
    df = pd.read_excel(xcel_fname)
    data_mask = pd.Series([True for _ in range(len(df))])
    for i in range(len(df)): #len(df)
        try:
            image = get_image(df.loc[i, "POINT_X"], df.loc[i, "POINT_Y"])
            save_image(image, image_folder, i)
        except:
            image = np.zeros((400,400,3))
            save_image(image, image_folder, i)

            missing.append(i)
            data_mask.iloc[i] = False
    print("Missing:", missing)
    data_mask.to_csv(image_mask_fname)
        
        

if __name__ == '__main__':
    image_folder = 'images/'
    excel_fname = "UrbFloodVul_Overall_StudyArea_Points.xlsx"
    image_mask_fname = "lost_images_mask.csv"
    main(image_folder, excel_fname, image_mask_fname )

#resp = urlopen('https://upload.wikimedia.org/wikipedia/commons/3/3e/Emperor_Penguins_(15885611526).jpg')
#resp = urlopen("https://maps.googleapis.com/maps/api/streetview?size=650x300&location=13.78848431,100.6040305&key=")
#image = np.asarray(bytearray(resp.read()), dtype='uint8')
#image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#image = cv2.resize(image, (int(image.shape[1]/5), int(image.shape[0]/5)))
