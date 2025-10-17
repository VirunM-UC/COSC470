import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import utils

LABELS = {"structure_type": ("attached", "semi-detached", "detached"),
          "building_conditions": ("very poor", "poor", "fair", "good", "very good"),
          "building_material": ("cinder", "brick")}


def print_metrics(df, attribute):
    LABEL2ID, ID2LABEL = dict(), dict()
    for i, label in enumerate(LABELS[attribute]):
        LABEL2ID[label] = str(i)
        ID2LABEL[str(i)] = label
    compute_metrics = utils.default_metric_maker(ID2LABEL)

    predictions = df[f"{attribute}_predict"].map(lambda x: [int(i == int(LABEL2ID[x])) for i in range(len(LABELS[attribute]))]).to_list() #one hot encoding
    labels = df[f"{attribute}_actual"].map(lambda x: int(LABEL2ID[x])).to_list()
    scores = compute_metrics((predictions, labels))
    print(", ".join(f"{key}: {value:.3f}" for key, value in scores.items()))

def main(data_folder, file_path, attribute, model_path, is_french = False):
    df = utils.load_data(data_folder, "validation.pkl")

    df = df[df[attribute].map(lambda x: (x in LABELS[attribute]))] #filter everything that is not in LABELS

    classifier = pipeline("image-classification", model = model_path)

    #This section would be functionalised with building_conditions as regression.
    scores = classifier(df["image"].to_list()) #result is list of list of dicts {score:, label:}. One dict for each label, one list of dicts for each image.
    max_scores = map(lambda per_image: max(per_image, key = lambda per_label: per_label['score']), scores) #result is list of dicts, each dict being th predicted label of an image
    attribute_predict = list(map(lambda per_image: per_image['label'], max_scores))

    if is_french:
        translation = {
            "beton": "cinder",
            "briques": "brick"
        }
        attribute_predict = list(map(lambda x: translation[x], attribute_predict))

    city_name = df.loc[:, "City_Name"]
    images = df.loc[:, "image"]
    point_x = df.loc[:, "POINT_X"]
    point_y = df.loc[:, "POINT_Y"]
    attribute_actual = df.loc[:, attribute]
    correct = [attribute_actual.iloc[i] == attribute_predict[i] for i in range(len(df))]

    df_attribute = pd.DataFrame({"City_Name"            : city_name, 
                                 "POINT_X"              : point_x,
                                 "POINT_Y"              : point_y,
                                 f"{attribute}_predict" : attribute_predict,
                                 f"{attribute}_actual"  : attribute_actual, 
                                  "correct"             : correct, 
                                  "image"               : images})

    utils.df_to_excel(df_attribute, file_path)

    #evaluate
    print("global - ", end='')
    print_metrics(df_attribute, attribute)
    for name in df_attribute["City_Name"].unique():
        city_df = df_attribute.loc[df_attribute["City_Name"] == name]
        print(f"{name} - ", end='')
        print_metrics(city_df, attribute)

    for region, cities in REGIONS.items():
        region_df = df_attribute.loc[df_attribute["City_Name"].map(lambda x: (x in cities))]
        print(f"{region} - ", end='')
        print_metrics(region_df, attribute)

    ConfusionMatrixDisplay.from_predictions(attribute_actual, attribute_predict, labels = LABELS[attribute])
    plt.show()


if __name__ == "__main__":
    REGIONS = {
        "europe": ["copenhagen", "sliven"],
        "north america": ["harriscounty", "britishcolumbia", "queretaro"],
        "south america": ["quito", "mocoa"],
        "australia": ["queensland"],
        "asia": ["dhaka", "phnompenh", "rajshahi", "bangkok", "jakarta", "kualalumpur", "manila"],
        "africa": ["durban"]
    }
    
    attribute = "building_material"
    #attribute = "structure_type"
    #attribute = "building_conditions"
    data_folder = "data-folders/material-data/" #material-data is the stratified global dataset for material
    model_path = f"model-folders/vit-building_material-model/checkpoint-63"
    file_path = f"excel-outputs/vit_{attribute}.xlsx"
    
    main(data_folder, file_path, attribute, model_path, is_french = False)