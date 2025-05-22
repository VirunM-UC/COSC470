import pandas as pd
from transformers import pipeline

def load_data(data_folder, file_name):
    """Takes a pickled dataframe and returns a Tensorflow Dataset
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


df = load_data("data/", "validation.pkl")

df = df[~df["structure_type"].isna()] #filter NaNs


classifier = pipeline("image-classification", model = "ViT-structure_type-model/checkpoint-78")

structure_type_predict = []
for i in range(len(df)):
    image = df.loc[df.index[i], "image"]
    scores = classifier(image) #result is list of dicts {score:, label:}
    max_score = max(scores, key = lambda x: x['score'])
    structure_type_predict.append(max_score['label'])

images = df.loc[:, "image"]
structure_type_actual = df.loc[:, "structure_type"]
correct = [structure_type_actual.iloc[i] == structure_type_predict[i] for i in range(len(df))]

df_structure_type = pd.DataFrame({"structure_type_predict": structure_type_predict, "structure_type_actual": structure_type_actual, 
                                  "correct": correct, "image": images})

df_to_excel(df_structure_type, "excel_outputs/vit_structure_type.xlsx")
