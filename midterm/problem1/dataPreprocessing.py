import numpy as np
import pandas as pd
from pathlib import Path
import time

from fiona.meta import extension


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper


class MidtermData:
    def __init(self, file_name, data_cat, thickness):
        self.file_name = file_name
        self.data_cat = data_cat
        self.thickness = thickness
        self.extension = Path(self.file_name).suffix.lower()
        self.file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.file_name}'

class FDCdata(MidtermData):
    def __init__(self, file_name, thickness):
        self.file_name = file_name
        self.data_cat = 'FDC_Data'
        self.thickness = thickness
        self.extension = Path(self.file_name).suffix.lower()
        self.file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.file_name}'
        print(f"FDC Data from {self.file_name} is created")

    @timer
    def get_data_frame(self):
        """
        :return: dataframe
        """

        extension = self.extension
        file_path = self.file_path

        if extension == '.csv':
            df = pd.read_csv(file_path, header=6)
            df = get_rid_of_space(df, 'ASE Cycles')
            print(f"data frame of {self.file_name} has created")
            return df
        elif extension == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=None, header=6)
            df = get_rid_of_space(df, 'ASE Cycles')
            print(f"data frame of {self.file_name} has created")
            return df
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

class OESdata(MidtermData):
    def __init__(self, file_name, thickness):
        self.file_name = file_name
        self.data_cat = 'OES_Data'
        self.thickness = thickness
        self.extension = Path(self.file_name).suffix.lower()
        self.file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.extension[1:]}/{self.file_name}'
        print(f"OES Data from {self.file_name} is created")

    @timer
    def get_data_frame(self):
        """
        :return: dataframe of file
        """

        extension = self.extension
        file_path = self.file_path

        if extension == '.csv':
            df = pd.read_csv(file_path, header=2)
            df = df.iloc[:,:3648]
            print(f"data frame of {self.file_name} has created")
            return df
        elif extension == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=1, header=2)
            print(f"data frame of {self.file_name} has created")
            return df
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

def get_rid_of_space(df, col_name):
    df[col_name] = df[col_name].replace(['    ', '   ', '  ', ' ', ''], np.nan)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)

    return df

def get_file_list(folder_path):
    # Get file name list of each FDC and OES Data
    folder_path = Path(folder_path)
    file_list = [f.name for f in folder_path.iterdir() if f.is_file()]
    print("File list has created.")
    print(file_list)

    return file_list

def __main__():

    thickness = '25um'
    oes_folder_path = f'./TSV_Etch_Dataset/{thickness}/OES_Data'
    oes_file_list = get_file_list(oes_folder_path)

    oes_df_list = []
    for i in range(1, len(oes_file_list)):
        oes_data = OESdata(oes_file_list[i], thickness)
        extension = Path(oes_file_list[i]).suffix.lower()
        if extension == '.csv':
            oes_df = oes_data.get_data_frame('csv')
            oes_df_list.append(oes_df)
            print(oes_df)
        elif extension == '.xlsx':
            oes_df = oes_data.get_data_frame('avg')
            oes_df_list.append(oes_df)
            print(oes_df)
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

if __name__ == "__main__":
    __main__()


