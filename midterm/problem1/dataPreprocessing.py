import numpy as np
import pandas as pd
from pathlib import Path
import re
import openpyxl
import time

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

    @timer
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

    @timer
    def get_cycle(self):

        extension = self.extension
        file_path = self.file_path

        if extension == '.xlsx':
            wb = openpyxl.load_workbook(file_path, data_only=False)
            avgs = wb.worksheets[1]
        else:
            raise ValueError(f'Unsupported File Format: {extension}')

        idx = 50
        cycle_range = []

        for row in avgs.iter_rows(min_col=1, max_col=1, values_only=False):
            cell = row[0]
            if cell.data_type == 'f':
                formula = cell.value
                start_cell, end_cell = extract_range_from_formula(formula)
                range_len = end_cell - start_cell
                cycle_range.append((idx, start_cell, end_cell, range_len))
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

def get_file_list(folder_path):
    # Get file name list of each FDC and OES Data
    folder_path = Path(folder_path)
    file_list = [f.name for f in folder_path.iterdir() if f.is_file()]
    print("File list has created.")
    print(file_list)

    return file_list

def get_range_data(oes_file_list, thickness):
    total_range_lst = []
    for oes_file in oes_file_list:
        oes_data = OESdata(oes_file, thickness)
        cycle_range_data = oes_data.get_cycle()
        cycle_range = []
        for cycle in cycle_range_data:
            cycle_range.append(cycle[3])
        total_range_lst.append(cycle_range)

    return total_range_lst

def __main__():

    thickness = '25um'
    oes_folder_path = f'./TSV_Etch_Dataset/{thickness}/OES_Data'
    oes_file_list = get_file_list(oes_folder_path)
    oes_xlsx_list = []
    for oes_file in oes_file_list:
        if '.xlsx' in oes_file:
            oes_xlsx_list.append(oes_file)
    print(oes_xlsx_list)
    total_cycle_range = get_range_data(oes_xlsx_list, thickness)
    print(total_cycle_range)

    avg_range = []
    for j in range(len(total_cycle_range[0])):
        for i in range(len(total_cycle_range)):
            cycle_sum = 0
            cycle_sum += total_cycle_range[i][j]
            avg_range.append(cycle_sum/len(total_cycle_range))
    print(avg_range)



    #fdc_file = "FDC_25um_Fault_C4F8_90to85_SCCM.CSV"
    #oes_file = "S2514.xlsx"

    #fdc_data = FDCdata(fdc_file, '25um')
    #oes_data = OESdata(oes_file, '25um')

    #fdc_df = fdc_data.get_data_frame()
    #oes_df = oes_data.get_data_frame()




if __name__ == "__main__":
    __main__()


