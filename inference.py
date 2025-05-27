import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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

def main(data_folder, file_path):
    df = load_data(data_folder, "validation.pkl")

    df = df[~df["structure_type"].isna()] #filter NaNs


    classifier = pipeline("image-classification", model = "vit-structure_type-model/checkpoint-78")

    scores = classifier(df["image"].to_list()) #result is list of list of dicts {score:, label:}. One dict for each label, one list of dicts for each image.
    max_scores = map(lambda per_image: max(per_image, key = lambda per_label: per_label['score']), scores) #result is list of dicts, each dict being th predicted label of an image
    structure_type_predict = list(map(lambda per_image: per_image['label'], max_scores))


    city_name = df.loc[:, "City_Name"]
    images = df.loc[:, "image"]
    structure_type_actual = df.loc[:, "structure_type"]
    correct = [structure_type_actual.iloc[i] == structure_type_predict[i] for i in range(len(df))]

    df_structure_type = pd.DataFrame({"City_Name": city_name, "structure_type_predict": structure_type_predict,
                                      "structure_type_actual": structure_type_actual, "correct": correct, "image": images})

    df_to_excel(df_structure_type, file_path)

    for name in df_structure_type["City_Name"].cat.categories:
        city_df = df_structure_type.loc[df_structure_type["City_Name"] == name]
        accuracy = sum(city_df["correct"]) / len(city_df)
        print(name, f"- Accuracy: {accuracy:.3f}")

    ConfusionMatrixDisplay.from_predictions(structure_type_actual, structure_type_predict, labels = ["attached", "semi-detached", "detached"])
    plt.show()


if __name__ == "__main__":
    data_folder = "data/"
    file_path = "excel_outputs/vit_structure_type.xlsx"
    main(data_folder, file_path)