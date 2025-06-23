import pandas as pd


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

    df_to_excel(df, file_path)


if __name__ == "__main__":
    data_folder = "data/"
    file_path = "excel-outputs/validate_set.xlsx"
    main(data_folder, file_path)