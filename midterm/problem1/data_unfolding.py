import dataPreprocessing as DP
import pandas as pd

def grouped_fdc(fdc_df):
    """
    get grouped FDC data from its df whose ASE Cycle value is not 0
    :param fdc_df: FDC df
    :return: grouped OES data in form of df
    """
    fdc_df = fdc_df[fdc_df['ASE Cycles'] != 0]
    fdc_grouped = [group for _, group in fdc_df.groupby("ASE Cycles")]
    fdc_grouped_sorted = sorted(fdc_grouped, key=lambda group: group["ASE Cycles"].iloc[0], reverse=True)

    print("Successfully Create Grouped FDC Data")
    return fdc_grouped_sorted

def grouped_oes(oes_df, oes_cycle):
    """
    grouping OES Data using all the Cycle range
    :param oes_df: OES df
    :param oes_cycle:
    :return:
    """
    oes_group_lst = []

    for cycle, start_cell_num, end_cell_num in oes_cycle:
        # extract the data from current range

        data = get_data_from_range(oes_df, start_cell_num, end_cell_num)

        # converting as DataFrame and adding Cycle label
        df = pd.DataFrame(data, columns=oes_df.columns)
        df["Cycle"] = cycle
        df = df.iloc[:, 1:]
        oes_group_lst.append(df)

    # Merging all cycle data
    concat_oes_df = pd.concat(oes_group_lst, ignore_index=True)

    oes_grouped = [group for _, group in concat_oes_df.groupby("Cycle")]
    oes_grouped_sorted = sorted(oes_grouped, key=lambda group: group["Cycle"].iloc[0], reverse=True)

    print("Successfully Create Grouped OES Data")

    return oes_grouped_sorted

def get_data_from_range(oes_df, start_cell_num, end_cell_num):
    """
    extracting data from the cell range of the specific sheet
    :param oes_df: OES df
    :param start_cell_num: start cell number of the range
    :param end_cell_num: end cell number of the range
    :return:
    """
    data = oes_df.iloc[start_cell_num - 1:end_cell_num + 1]
    return data

def unfolding_df(df):
    row_vector = df.values.flatten()
    merged_row_vector = pd.DataFrame([row_vector])
    return merged_row_vector

def get_unfolded_x_data(fdc_file, oes_file):
    """
    make unfolded X data (1 row) merging FDC data and OES Data
    :param fdc_file: FDC file name
    :param oes_file: OES file name
    :return: 1 row vector of unfolded X data
    """

    fdc_data = DP.FDCdata(fdc_file, '25um')
    oes_data = DP.OESdata(oes_file, '25um')

    fdc_df = fdc_data.get_data_frame()

    if oes_data.extension == '.csv':
        oes_df = oes_data.get_data_frame()
    elif oes_data.extension == '.xlsx':
        oes_df = oes_data.get_data_frame()
        oes_cycle = oes_data.get_cycle()
    else:
        raise ValueError(f"Unsupported file extension {oes_data.extension}")

    fdc_grouped = grouped_fdc(fdc_df)
    oes_grouped = grouped_oes(oes_df, oes_cycle)

    fdc_row_vector = pd.DataFrame()
    for group in fdc_grouped:
        cycle_row_vector = unfolding_df(group)
        fdc_row_vector = pd.concat([fdc_row_vector, cycle_row_vector], ignore_index=True)
    fdc_unfolded = unfolding_df(fdc_row_vector)

    oes_row_vector = pd.DataFrame()
    for group in oes_grouped:
        cycle_row_vector = unfolding_df(group)
        oes_row_vector = pd.concat([oes_row_vector, cycle_row_vector], ignore_index=True)
    oes_unfolded = unfolding_df(oes_row_vector)

    merged_data = pd.concat([fdc_unfolded, oes_unfolded], ignore_index=True)
    flatten_data = merged_data.values.flatten()

    unfolded_x_data = pd.DataFrame([flatten_data])

    print(f"Successfully Create unfolded X data with shape of {unfolded_x_data.shape}")
    print(unfolded_x_data)

    return unfolded_x_data


if __name__ == "__main__":
    fdc_file = "FDC_TSV25_(14).CSV"
    oes_file = "S2514.xlsx"

    get_unfolded_x_data(fdc_file, oes_file)