import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import utils

LABELS = {"structure_type": ("attached", "semi-detached", "detached"),
          "building_conditions": ("very poor", "poor", "fair", "good", "very good")}


def main(data_folder, file_path, attribute, model_path):
    df = utils.load_data(data_folder, "validation.pkl")

    df = df[~df[attribute].isna()] #filter NaNs


    classifier = pipeline("image-classification", model = model_path)

    #This section would be functionalised with building_conditions as regression.
    scores = classifier(df["image"].to_list()) #result is list of list of dicts {score:, label:}. One dict for each label, one list of dicts for each image.
    max_scores = map(lambda per_image: max(per_image, key = lambda per_label: per_label['score']), scores) #result is list of dicts, each dict being th predicted label of an image
    attribute_predict = list(map(lambda per_image: per_image['label'], max_scores))


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

    accuracy = sum(df_attribute["correct"]) / len(df_attribute)
    print(f"{attribute} - Accuracy: {accuracy:.3f}")
    for name in df_attribute["City_Name"].cat.categories:
        city_df = df_attribute.loc[df_attribute["City_Name"] == name]
        accuracy = sum(city_df["correct"]) / len(city_df)
        print(name, f"- Accuracy: {accuracy:.3f}")

    ConfusionMatrixDisplay.from_predictions(attribute_actual, attribute_predict, labels = LABELS[attribute])
    plt.show()


if __name__ == "__main__":
    attribute = "structure_type"
    #attribute = "building_conditions"
    data_folder = "data-folders/data/"
    model_path = "model-folders/vit-structure_type-model/checkpoint-78"
    file_path = f"excel-outputs/vit_{attribute}.xlsx"
    main(data_folder, file_path, attribute, model_path)