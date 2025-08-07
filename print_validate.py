import pandas as pd
import utils

def main(data_folder, file_path):
    df = utils.load_data(data_folder, "validation.pkl")

    utils.df_to_excel(df, file_path)


if __name__ == "__main__":
    data_folder = "data-folders/french-data/"
    file_path = "excel-outputs/validate_set.xlsx"
    main(data_folder, file_path)