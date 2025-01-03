from pathlib import Path
import pandas as pd
import data_unfolding
from PCA import DynamicPCA
from LOOCV import LOOCV
from kernelPCA import DynamicKernelPCA
from PCR import PCR
import os
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

def get_file_list(folder_path):
    # Get file name list of each FDC and OES Data
    folder_path = Path(folder_path)
    file_list = [f.name for f in folder_path.iterdir() if f.is_file()]
    print("File list has created.")
    print(file_list)

    return file_list

def get_input_data(fdc_file_list, oes_file_list, thickness):
    # Make X as Data Frame of 17 dataset
    X = pd.DataFrame()
    for fdc_file, oes_file in zip(fdc_file_list, oes_file_list):
        unfolded_fdc = data_unfolding.unfolding_fdc(fdc_file, thickness)
        unfolded_oes = data_unfolding.unfolding_oes(oes_file, thickness)
        unfolded_data = data_unfolding.get_unfolded_x_data(unfolded_fdc, unfolded_oes)
        X = pd.concat([X, unfolded_data])
    output_path = f'./processed_data/{thickness}/X_data.csv'
    X.to_csv(output_path, index=False)
    return X

def get_depth(thickness):
    """
    get Series of Depth from TSV_Depth_new.xlsx, given thickness
    :param thickness: which thickness Depth from
    :return: Series of Depth
    """

    file_path = './TSV_Etch_Dataset/TSV_Depth_new.xlsx'
    depth_df = pd.read_excel(file_path, sheet_name='TSV depth')
    depth_df.columns = depth_df.columns.str.replace(r'\s+', '', regex=True)
    success_depth_col = depth_df.columns[depth_df.columns.get_loc(thickness) + 1]
    success_depth = depth_df[success_depth_col][1:13]
    failed_depth = depth_df[f"Fault{thickness}"][1:6]
    full_depth = pd.concat([failed_depth, success_depth], ignore_index=True)

    return full_depth

@timer
def normalize(X):
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.loc[:, X.std() != 0]
    std = X.std() + 1e-8
    X_normalized = X / std

    return X_normalized

def get_formatted_y(thickness):
    y = get_depth(thickness)
    y = y.squeeze()
    y.name = 'Depth'
    y = pd.to_numeric(y, errors='coerce')

    return y

@timer
def dynamicPCA(X, target_variance):
    pca = DynamicPCA(target_variance)
    data_reduced = pca.fit_transform(X)
    print(f"Reduced Data Size : {data_reduced.shape}")

    pca.plot_scree()
    pca.plot_score(pc1=1, pc2=2)
    pca.plot_loading(pc=1)
    return data_reduced, pca

@timer
def dynamicKernelPCA(X, target_variance):
    kpca = DynamicKernelPCA(kernel='poly', gamma=15, target_variance=0.9)
    data_reduced = kpca.fit_transform(X)
    print(f"Reduced Data Size : {data_reduced.shape}")

    kpca.visualizing()
    return data_reduced, kpca


def __main__():
    thickness = '25um'

    if 'X_data.csv' in os.listdir(f'./processed_data/{thickness}'):
        X = pd.read_csv(f'./processed_data/{thickness}/X_data.csv').fillna(0)
    else:
        fdc_folder_path = f'./TSV_Etch_Dataset/{thickness}/FDC_Data'
        oes_folder_path = f'./TSV_Etch_Dataset/{thickness}/OES_Data/csv'

        fdc_file_list = get_file_list(fdc_folder_path)
        oes_file_list = get_file_list(oes_folder_path)

        X = get_input_data(fdc_file_list, oes_file_list, thickness).fillna(0)

    print(X)
    X = normalize(X)
    print(X)

    y = get_formatted_y(thickness)
    print(y)

    X_pca, pca = dynamicPCA(X, target_variance=0.9)
    print(f"PCA Data Reduced : {X_pca}")
    print(f"PCA Data Reduced Shape : {X_pca.shape}")

    X_kpca, kpca = dynamicKernelPCA(X, target_variance=0.9)
    print(f"Kernel PCA Data Reduced with Poly : {X_kpca}")
    print(f"Kernel PCA Data Reduced Shape Poly : {X_kpca.shape}")

    pca_pcr = PCR()
    pca_pcr.fit(X_pca, y)
    pca_coefficient, pca_intercept = pca_pcr.get_coefficients()
    print(f"Regression Coefficient with PCA : {pca_coefficient}")
    print(f"Regression Intercept with PCA : {pca_intercept}")

    kpca_pcr = PCR()
    kpca_pcr.fit(X_kpca, y)
    kpca_coefficient, kpca_intercept = kpca_pcr.get_coefficients()
    print(f"Regression Coefficient with KPCA : {kpca_coefficient}")
    print(f"Regression Intercept with KPCA : {kpca_intercept}")

    metrics = ['mse', 'mape']
    for metric in metrics:
        loocv_pca = LOOCV(pca_pcr)
        loocv_pca.fit(X_pca, y)
        print(f"LOOCV of PCR with PCA using {metric.upper()} : {loocv_pca.mean_score(metric):.4f}")
    print(f"PCR with PCA using R² : {pca_pcr.score(X_pca, y)[2]}")

    for metric in metrics:
        loocv_kpca = LOOCV(kpca_pcr)
        loocv_kpca.fit(X_kpca, y)
        print(f"LOOCV of PCR with KPCA using {metric.upper()} : {loocv_kpca.mean_score(metric):.4f}")
    print(f"PCR with KPCA using R² : {kpca_pcr.score(X_kpca, y)[2]}")

if __name__ == '__main__':
    __main__()
