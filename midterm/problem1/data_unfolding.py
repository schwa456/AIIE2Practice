import dataPreprocessing as DP
import pandas as pd

def grouped_fdc(fdc_df):
    """
    get grouped FDC data from its df
    :param fdc_df: FDC df
    :return: grouped OES data in form of df
    """
    fdc_grouped = [group for _, group in fdc_df.groupby("ASE Cycles")]
    fdc_grouped_sorted = sorted(fdc_grouped, key=lambda group: group["ASE Cycles"].iloc[0], reverse=True)
    return fdc_grouped_sorted

def grouped_oes(oes_df, cycle_range):
    """
    grouping OES Data using all the Cycle range
    :param oes_df: OES df
    :param cycle_range:
    :return:
    """
    grouped_oes = []

    for cycle, start_cell_num, end_cell_num in cycle_range:
        # extract the data from current range

        data = get_data_from_range(oes_df, start_cell_num, end_cell_num)

        # converting as DataFrame and adding Cycle label
        df = pd.DataFrame(data, columns=oes_df.columns)
        df["Cycle"] = cycle
        grouped_oes.append(df)

    # Merging all cycle data
    final_grouped_oes = pd.concat(grouped_oes, ignore_index=True)

    return final_grouped_oes

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


def __main__():
    fdc_file = "FDC_TSV25_(14).CSV"
    oes_file = "S2514.xlsx"

    fdc_data = DP.FDCdata(fdc_file, '25um')
    oes_data = DP.OESdata(oes_file, '25um')

    fdc_df = fdc_data.get_data_frame()
    if oes_data.extension == '.csv':
        oes_df = oes_data.get_data_frame()
    elif oes_data.extension == '.xlsx':
        oes_df = oes_data.get_data_frame()
        oes_cycle = oes_data.get_cycle()
        print(oes_cycle)
    else:
        raise ValueError(f"Unsupported file extension {oes_data.extension}")

    fdc_grouped = grouped_fdc(fdc_df)
    oes_grouped = grouped_oes(oes_df, oes_cycle)

    print(fdc_grouped)
    print(oes_grouped)

if __name__ == "__main__":
    __main__()