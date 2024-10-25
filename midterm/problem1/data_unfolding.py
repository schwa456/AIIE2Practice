import dataPreprocessing as dp
import pandas as pd
import openpyxl
import re

def grouped_fdc(fdc_df):
    fdc_grouped = [group for _, group in fdc_df.groupby("ASE Cycles")]
    fdc_grouped_sorted = sorted(fdc_grouped, key=lambda group: group["ASE Cycles"].iloc[0], reverse=True)
    return fdc_grouped_sorted

def extract_cycle_range(range_sheet_name):
    """
    parsing range_sheet to extract each cycle's range
    :param range_sheet_name: name of the range_sheet to parse
    :return: cycle_range
    """

    # initializing cycle range
    cycle_range = []

    for row in range_sheet_name.iter_rows(min_row=2, max_col=2, values_only=True):
        cycle, formula = row
        sheet_name, start_cell, end_cell = extract_range_from_formula(formula)
        cycle_range.append((sheet_name, start_cell, end_cell))
        print(f"Extracted all the values of cycle {cycle} with length of {len(cycle_range)}")

    return cycle_range

def extract_range_from_formula(formula):
    """
    extract sheet name and cell range from formula
    :param formula: the formula that contains the range
    :return: (sheet_name, start_cell, end_cell)
    """

    match = re.search(r"'(\d+)'!\$?([A-Z]+\d+):\$?([A-Z]+\d+)", formula)
    if match:
        print(f"Successfully Extracted Cycle : {match.groups()[0], match.groups[1], match.groups[2]}")
        return match.groups()
    return None, None, None


def grouped_oes(all_sheets, cycle_range):
    """
    grouping OES Data using all the Cycle range
    :param all_sheets:
    :param cycle_range:
    :return:
    """
    grouped_oes = []

    for cycle, sheet_name, start_cell, end_cell in cycle_range:
        # extract the data from current range
        sheet = all_sheets[sheet_name]
        data = get_data_from_range(sheet, start_cell, end_cell)

        # converting as DataFrame and adding Cycle label
        df = pd.DataFrame(data, columns=[f"Cycle_{cycle}}_Data"])
        df["Cycle"] = cycle
        grouped_oes.append(df)

    # Merging all cycle data
    final_grouped_oes = pd.concat(grouped_oes, ignore_index=True)

    return final_grouped_oes

def get_data_from_range(sheet, start_cell, end_cell):
    """
    extracting data from the cell range of the specific sheet
    :param sheet: the specific sheet that contains the range
    :param start_cell: start cell of the range
    :param end_cell: end cell of the range
    :return:
    """

    cells = sheet[start_cell:end_cell]
    return [[cell.value for cell in row if cell.value is not None] for row in cells]


def extract_all_oes_data(oes_sheet):
    """
    extract all the values in the range of parsed result of OES data's formula
    :param oes_sheet: sheet where the formulas are contained
    :return: all_cycle_data as list
    """

    sheet1 = oes_sheet['Sheet1']
    all_cycle_data = []

    for row in sheet1.iter_rows(min_row=2, max_col=2, values_only=True):
        cycle, formula = row
        print(f"Cycle : {cycle}, Formula : {formula}")
        sheet_name, start_cell, end_cell = extract_cycle_from_data(formula)

        # extract all the values of range of each cycle
        cycle_data = get_all_values_from_range(oes_sheet[sheet_name], start_cell, end_cell)
        cycle_df = pd.DataFrame(cycle_data, columns=[f"Cycle_{cycle}_Data"])
        print(f"Extracted all the values of cycle {cycle} with shape of {cycle_df.shape}")
        all_cycle_data.append(cycle_df)

    print(f"Successfully Extracted all the values of all the cycle with shape of {all_cycle_data.shape}")
    return all_cycle_data


def extract_cycle_from_data(formula):
    """
    :param formula: formula from OES excel data, which contains range of cycle
    :return:
    """

    match = re.search(r"'(\d+)'!\$?([A-Z]+\d+):\$?([A-Z]+\d+)", formula)
    if match:
        print(f"Successfully Extracted Cycle : {match.groups()[0], match.groups[1], match.groups[2]}")
        return match.groups()
    return None, None, None


def get_all_values_from_range(sheet, start_cell, end_cell):
    """
    extract all the values of sheet from start cell to end cell
    :param sheet: the sheet where values are extracted
    :param start_cell: start cell where values are extracted
    :param end_cell: end cell where values are extracted
    :return: list of all values from start_cell to end_cell
    """

    cells = sheet[start_cell:end_cell]
    values = [[cell.value for cell in row if cell.value is not None] for row in cells]
    print(f"Extracted All values from {start_cell} to {end_cell}, whose length is {len(values)}")
    return values

def __main__():
    fdc_file = "FDC_TSV25_(14).CSV"
    oes_file = "S2514.xlsx"

    fdc_data = dp.FDCdata(fdc_file)
    oes_data = dp.OESdata(oes_file)

    fdc_df = fdc_data.get_data_frame('25um')
    oes_df = oes_data.get_data_frame('25um')

    fdc_grouped = grouped_fdc(fdc_df)
    print(fdc_grouped)



if __name__ == "__main__":
    __main__()