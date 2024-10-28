import numpy as np
import pandas as pd
from pathlib import Path
import re


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

    def get_data_frame(self):
        """
        :param thickness: thickness of data which get dataframe from
        :return: dataframe
        """

        extension = Path(self.file_name).suffix.lower()
        file_path = f'./TSV_Etch_Dataset/{self.thickness}/{self.data_cat}/{self.file_name}'

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
        self.file_path = f'./TSV_Etch_Dataset/{thickness}/{self.data_cat}/{self.file_name}'
        print(f"OES Data from {self.file_name} is created")


    def get_data_frame(self):
        """
        :return: dataframe of file
        """

        extension = self.extension
        file_path = self.file_path

        if extension == '.csv':
            df = pd.read_csv(file_path, header=2)
            print(f"data frame of {self.file_name} has created")
            return df
        elif extension == '.xlsx':
            df = pd.read_excel(file_path, sheet_name=0, header=2)
            print(f"data frame of {self.file_name} has created")
            return df
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

    def get_cycle(self):
        sheet = self.wb['Sheet1']
        idx = 50
        cycle_range = []

        for row in sheet.iter_rows(min_row=1, min_col=1, max_col=1):
            for cell in row:
                if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                    formula = cell.value
                    start_cell, end_cell = extract_range_from_formula(formula)
                    cycle_range.append((idx, start_cell, end_cell))
                    idx -= 1

        print(f"Successfully get cycle of {self.file_name}")
        return cycle_range


def get_rid_of_space(df, col_name):
    df[col_name] = df[col_name].replace(['    ', '   ', '  ', ' ', ''], np.nan)
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)

    return df

def extract_range_from_formula(formula):
    match = re.search(r"'(\d+)'!\$?[A-Z]+(\d+):\$?[A-Z]+(\d+)", formula)
    if match:
        return int(match.groups()[1]), int(match.groups()[2])
    return None, None


if __name__ == "__main__":
    fdc_file = "FDC_25um_Fault_C4F8_90to85_SCCM.CSV"
    oes_file = "OES_25um_Fault_C4F8_90to85_SCCM.csv"

    fdc_data = FDCdata(fdc_file, '25um')
    oes_data = OESdata(oes_file, '25um')

    fdc_df = fdc_data.get_data_frame()
    oes_df = oes_data.get_data_frame()
