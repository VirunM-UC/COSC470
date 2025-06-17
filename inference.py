import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = {"structure_type": ("attached", "semi-detached", "detached"),
          "building_conditions": ("very poor", "poor", "fair", "good", "very good")}

def load_data(data_folder, file_name):
    """Takes a pickled dataframe and returns a Pandas Dataframe
    """
    df = pd.read_pickle(data_folder + file_name)
    return df


def df_to_excel(df, file_path):
    df = df.iloc[:, :-1]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Get the xlsxwriter workbook and worksheet objects.
    worksheet = writer.sheets['Sheet1']

    # Insert an image.
    for i in range(len(df)):
        worksheet.insert_image(f'P{i+2}', f'images/image_{df.index[i]}.jpg')

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()

def main(data_folder, file_path, attribute, checkpoint_num):
    df = load_data(data_folder, "validation.pkl")

    df = df[~df[attribute].isna()] #filter NaNs


    classifier = pipeline("image-classification", model = f"vit-{attribute}-model/checkpoint-{checkpoint_num}")

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

    df_to_excel(df_attribute, file_path)

    accuracy = sum(df_attribute["correct"]) / len(df_attribute)
    print(f"{attribute} - Accuracy: {accuracy:.3f}")
    for name in df_attribute["City_Name"].cat.categories:
        city_df = df_attribute.loc[df_attribute["City_Name"] == name]
        accuracy = sum(city_df["correct"]) / len(city_df)
        print(name, f"- Accuracy: {accuracy:.3f}")

    ConfusionMatrixDisplay.from_predictions(attribute_actual, attribute_predict, labels = LABELS[attribute])
    plt.show()


if __name__ == "__main__":
    attribute, checkpoint_num = "structure_type", 78
    #attribute, checkpoint_num = "building_conditions", 75
    data_folder = "data/"
    file_path = f"excel_outputs/vit_{attribute}.xlsx"
    main(data_folder, file_path, attribute, checkpoint_num)