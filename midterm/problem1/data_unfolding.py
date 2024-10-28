import dataPreprocessing as DP
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

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

@timer
def unfolding_fdc(fdc_file, thickness):
    """
    make unfolded FDC df(1 row)
    :param fdc_file: file name of FDC data
    :param thickness: thickness
    :return: unfolded FDC df (1 row)
    """

    fdc_data = DP.FDCdata(fdc_file, thickness)

    fdc_df = fdc_data.get_data_frame()

    fdc_df.columns = fdc_df.columns.str.strip()
    # filtering only the columns whose ASE Cycles column has actual value
    if 'ASE Cycles' in fdc_df.columns:
        fdc_df = fdc_df[fdc_df['ASE Cycles'] != 0]
    else:
        print("Column named 'ASE Cycles' Does Not Exist")

    # select numeric columns
    all_features = fdc_df.columns.tolist()
    fdc_df = fdc_df[all_features[0:19]]

    fdc_row_vector = fdc_df.values.flatten()

    fdc_col_names = [f"{col}_{i}" for i in range(fdc_df.shape[0]) for col in fdc_df.columns]

    fdc_flattened_data = pd.DataFrame([fdc_row_vector], columns=fdc_col_names)
    print("FDC Flattened Data")
    print(fdc_flattened_data)

    return fdc_flattened_data

@timer
def unfolding_oes(oes_file, thickness):
    """
    make unfolded OES df(1 row)
    :param oes_file: file name of OES data
    :param thickness: thickness
    :return: unfolded OES df (1 row)
    """
    oes_data = DP.OESdata(oes_file, thickness)

    oes_df = oes_data.get_data_frame()
    oes_df = oes_df.loc[:, ~oes_df.columns.str.startswith('Unnamed')]

    # select numeric columns
    all_features = oes_df.columns.tolist()
    oes_df = oes_df[all_features[1:]]

    oes_row_vector = oes_df.values.flatten()
    print("OES row vector has been created.")
    print(oes_row_vector)

    oes_row_vector_2d = oes_row_vector.reshape(1, -1)

    oes_col_names = [f"{col}_{i}" for i in range(oes_df.shape[0]) for col in oes_df.columns]
    print("OES Columns names have been determined.")

    n_partition = 5
    chunk_size = oes_row_vector_2d.shape[1] // n_partition
    remainder = oes_row_vector_2d.shape[1] % n_partition
    print(f"Chunk size is {chunk_size}, remaining is {remainder}")

    chunks = [(1, ) + (chunk_size, )] * (n_partition - 1) + [(chunk_size + remainder, )]
    print(f"Chunks : {chunks}")

    dask_array = [
        da.from_array(
            oes_row_vector_2d[:, i * chunk_size : (i+1) * chunk_size],
            chunks=(1, chunk_size),
            )
            for i in range(n_partition - 1)
        ]
    dask_array.append(
        da.from_array(
            oes_row_vector_2d[:, (n_partition - 1) * chunk_size : ],
            chunks = (1, chunk_size + remainder)
        )
    )
    dask_array = da.concatenate(dask_array, axis=1)
    print("Dask Array has been created")
    print(f"Dask Array Shape : {dask_array.shape}, Chunks : {dask_array.chunks}")

    ddf = dd.from_dask_array(dask_array, columns=oes_col_names)
    print("Dask Data Frame has been created")

    oes_flattened_data = ddf.compute()

    print("OES Flattened Data")
    print(oes_flattened_data)


def get_unfolded_x_data(unfolded_fdc, unfolded_oes):
    """
    make unfolded X data (1 row) merging FDC data and OES Data
    :param unfolded_fdc: unfolded FDC data
    :param unfolded_oes: unfolded OES data
    :return: 1 row vector of unfolded X data
    """

    merged_data = pd.concat([unfolded_fdc, unfolded_oes], ignore_index=True)
    flatten_data = merged_data.values.flatten()

    unfolded_x_data = pd.DataFrame([flatten_data])

    print(f"Successfully Create unfolded X data with shape of {unfolded_x_data.shape}")
    print(unfolded_x_data)

    return unfolded_x_data


if __name__ == "__main__":
    fdc_file = "FDC_TSV25_(14).CSV"
    oes_file = "S2514.xlsx"

    get_unfolded_x_data(fdc_file, oes_file)