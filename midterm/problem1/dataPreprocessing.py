import numpy as np
import pandas as pd
from pathlib import Path

class MidtermData:
    def __init(self, file_name, data_cat):
        self.file_name = file_name
        self.data_cat = data_cat

class FDCdata(MidtermData):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_cat = 'FDC_Data'
        print(f"FDC Data from {self.file_name} is created")

    def get_data_frame(self, thickness):
        """
        :param thickness: thickness of data which get dataframe from
        :return: dataframe
        """

        extension = Path(self.file_name).suffix.lower()
        file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.file_name}'

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
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_cat = 'OES_Data'
        print(f"OES Data from {self.file_name} is created")

    def get_data_frame(self, thickness):
        """
        :param thickness: thickness of data which get dataframe from
        :return: dataframe
        """

        extension = Path(self.file_name).suffix.lower()
        file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.file_name}'

        if extension == '.csv':
            df = pd.read_csv(file_path, header=2)
            print(f"data frame of {self.file_name} has created")
            return df
        elif extension == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=None, header=2)
            print(f"data frame of {self.file_name} has created")
            return df
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

def get_rid_of_space(df, col_name):
    df[col_name] = df[col_name].replace(['    ', '   ', '  ', ' ', ''], np.nan)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)

    return df


if __name__ == "__main__":
    fdc_file = "FDC_25um_Fault_C4F8_90to85_SCCM.CSV"
    oes_file = "OES_25um_Fault_C4F8_90to85_SCCM.csv"

    fdc_data = FDCdata(fdc_file)
    oes_data = OESdata(oes_file)

    fdc_df = fdc_data.get_data_frame('25um')
    oes_df = oes_data.get_data_frame('25um')
