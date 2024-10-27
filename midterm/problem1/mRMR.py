import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dataPreprocessing import FDCdata
import data_unfolding
from sklearn.metrics import mutual_info_score


def mRMR(X, y, num_features):
    """
    mRMR 알고리즘을 사용하여 특징을 선택하는 함수

    Parameters:
    - X: 2차원 dataframe (샘플 수 x 특징 수) - 입력 특징 행렬
    - y: 1차원 dataframe 혹은 Series - 대상 변수 (레이블)
    - num_features: 선택할 특징의 수

    Returns:
    - selected_features: 선택된 특징의 인덱스 리스트
    """

    # 변수 초기화
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))

    # 상호정보량 계산
    relevance = np.array([mutual_info_score(X[:, i], y) for i in range(n_features)])

    # 첫번째 feature 선택(최대 관련성 기준)
    first_feature = np.argmax(relevance)
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # 남은 특징에서 반복적으로 특징 선택
    for _ in range(num_features - 1):
        max_score = -np.inf
        best_feature = -1

        # 각 남은 특징에 대해 mRMR 계산
        for feature in remaining_features:
            redundancy = np.mean([mutual_info_score(X[:, feature], X[:, f]) for f in selected_features])
            score = relevance[feature] - redundancy

            # 최대 점수를 가진 특징 선택
            if score > max_score:
                max_score = score
                best_feature = feature

        # 선택된 특징 update
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    return selected_features

def get_file_list(folder_path):
    # Get file name list of each FDC and OES Data
    folder_path = Path(folder_path)
    file_list = [f.name for f in folder_path.iterdir() if f.is_file()]
    file_list = file_list[5:]

    return file_list

def get_input_data(fdc_file_list, oes_file_list, output_path):
    # Make X as Data Frame of 12 successful dataset
    X = pd.DataFrame()
    for fdc_file, oes_file in zip(fdc_file_list, oes_file_list):
        unfolded_data = get_test_unfolded_x_data(fdc_file)
        X = pd.concat([X, unfolded_data])
    print(X)
    X.to_csv(output_path, index=False)
    return X

def get_test_unfolded_x_data(fdc_file):
    """
    make unfolded X data (1 row) merging FDC data and OES Data
    :param fdc_file: FDC file name
    :return: 1 row vector of unfolded X data
    """

    fdc_data = FDCdata(fdc_file, '25um')

    fdc_df = fdc_data.get_data_frame()

    fdc_grouped = data_unfolding.grouped_fdc(fdc_df)

    fdc_row_vector = pd.DataFrame()
    for group in fdc_grouped:
        cycle_row_vector = data_unfolding.unfolding_df(group)
        fdc_row_vector = pd.concat([fdc_row_vector, cycle_row_vector], ignore_index=True)
    fdc_unfolded = data_unfolding.unfolding_df(fdc_row_vector)

    unfolded_x_data = pd.DataFrame(fdc_unfolded)

    print(f"Successfully Create unfolded X data with shape of {unfolded_x_data.shape}")
    print(unfolded_x_data)

    return unfolded_x_data


def get_test_input_data(fdc_file_list):
    # Make X as Data Frame of 12 successful FDC dataset
    X = pd.DataFrame()
    for fdc_file in fdc_file_list:
        unfolded_data = get_test_unfolded_x_data(fdc_file)
        X = pd.concat([X, unfolded_data])

    return X

def __main__():
    folder_path = './TSV_Etch_Dataset/25um/FDC_Data'
    fdc_file_list = get_file_list(folder_path)
    X = get_test_input_data(fdc_file_list)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(X)
    print(X.shape)

    y = pd.read_excel('./TSV_Etch_Dataset/TSV_Depth_new.xlsx', usecols=[5], skiprows=2, nrows=12, header=None)
    y = y.squeeze()
    y = pd.to_numeric(y, errors='coerce')
    print(y)
    print(y.shape)



    print("mRMR result")



