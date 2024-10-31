import dataPreprocessing as DP
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

@timer
def get_cycle_range_by_FFT(df, file_name):
    # 시간 데이터를 초 단위로 변환
    df['Time_in_seconds'] = df['Time(sec)'].apply(time_to_second)
    time = df['Time_in_seconds'].values

    # 각 시간에 대한 모든 파장의 평균 강도 계산
    intensity_avg = df.iloc[:, 1:-1].mean(axis=1).values  # 모든 파장 열의 평균 강도

    # FFT 수행
    intensity_avg -= np.mean(intensity_avg)  # DC 성분 제거
    fft_result = np.fft.fft(intensity_avg)
    fft_freq = np.fft.fftfreq(len(intensity_avg), d=np.mean(np.diff(time)))

    # 주파수 스펙트럼 시각화
    magnitude = np.abs(fft_result)
    half_len = len(intensity_avg) // 2

    # 올바른 주파수를 찾기 위해 DC 성분 제외하고 최대값 찾기
    peak_freq = fft_freq[1:half_len][np.argmax(magnitude[1:half_len]) + 1]
    period = 1 / peak_freq if peak_freq != 0 else float('inf')
    period = period * 2

    # 주기가 너무 큰지 체크하고, 데이터 범위에 맞게 50번 반복
    max_time = time[-1]  # 전체 데이터의 시간 범위 끝
    if period * 50 > max_time:
        print(f"계산된 주기가 데이터의 전체 길이보다 길게 나옵니다. 주기: {period} 초")
        return peak_freq, period, None

    # 50번 반복 주기에 해당하는 주기 점선 위치 계산
    cycle_points = [time[0] + period * i for i in range(1, 51) if period * i <= max_time]

    # 평균 강도 시각화 및 주기별 점선 표시
    plt.figure(figsize=(10, 6))
    plt.plot(time, intensity_avg, 'b-', label="Average Intensity")

    # 주기별로 수직 점선 그리기
    for cp in cycle_points:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7)

    plt.xlabel("Time (sec)")
    plt.ylabel("Average Intensity (A.U.)")
    plt.title(f"Average Intensity over Time with Repeated Cycles with file name {file_name}")
    plt.legend(["Average Intensity", "Cycle Intervals"])
    plt.grid(True)
    plt.show()

    df['Cycle_group'] = (df['Time_in_seconds'] // period).astype(int)

    df = df.drop(columns=['Time(sec)', 'Time_in_seconds'])

    return df

def time_to_second(time_str):
    hours, minutes, seconds = 0, 0, 0
    if 'h' in time_str:
        hours = int(time_str.split('h')[0])
        time_str = time_str.split('h')[1]
    if 'm' in time_str:
        minutes = int(time_str.split('m')[0])
        time_str = time_str.split('m')[1]
    if 's' in time_str:
        seconds = int(time_str.split('s')[0])
    return hours * 3600 + minutes * 60 + seconds

def grouped_fdc(fdc_df):
    """
    get grouped FDC data from its df whose ASE Cycle value is not 0
    :param fdc_df: FDC df
    :return: grouped FDC data in form of df
    """
    fdc_df = fdc_df[fdc_df['ASE Cycles'] != 0]
    fdc_df = fdc_df.iloc[:, 1:22]
    fdc_df.drop(columns=['Process Phase', 'ASE Phase'], inplace=True)
    fdc_grouped = fdc_df.groupby("ASE Cycles")

    print("Successfully Create Grouped FDC Data")
    return fdc_grouped

def grouped_oes(oes_df, oes_file):
    """
    grouping OES Data using all the Cycle range
    :param oes_df: OES df
    :return:
    """

    oes_fft_df = get_cycle_range_by_FFT(oes_df, oes_file)
    oes_grouped_df = oes_fft_df.groupby('Cycle_group')

    print("Successfully Create Grouped OES Data")
    return oes_grouped_df

@timer
def unfolding_df(grouped_df):
    mean_group_df = grouped_df.mean()
    col_names = [f"{col}_{i}" for i in range(mean_group_df.shape[0]) for col in mean_group_df.columns]
    flattened_value = mean_group_df.values.flatten()
    merged_row_df = pd.DataFrame([flattened_value], columns=col_names)

    return merged_row_df

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

    fdc_grouped = grouped_fdc(fdc_df)

    unfolded_fdc = unfolding_df(fdc_grouped)
    return unfolded_fdc

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

    oes_grouped = grouped_oes(oes_df, oes_file)

    unfolded_oes = unfolding_df(oes_grouped)
    return unfolded_oes

@timer
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
    thickness = '25um'

    unfolded_fdc = unfolding_fdc(fdc_file, thickness)
    unfolded_oes = unfolding_oes(oes_file, thickness)

    get_unfolded_x_data(unfolded_fdc, unfolded_oes)
