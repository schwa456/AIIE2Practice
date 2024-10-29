from randomForest import RandomForestRegressor
from pathlib import Path
import pandas as pd
import data_unfolding
from LOOCV import LOOCV


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
    output_path = f'./processed_data/{thickness}/X_data_2.csv'
    X.to_csv(output_path, index=False)
    return X

def get_input_fdc_data(fdc_file_list, thickness):
    X = pd.DataFrame()
    for fdc_file in fdc_file_list:
        unfolded_fdc = data_unfolding.unfolding_fdc(fdc_file, thickness)
        X = pd.concat([X, unfolded_fdc])
    output_path = f'./processed_data/{thickness}/FDC_Data/X_fdc_data_2.csv'
    X.to_csv(output_path, index=False)
    return X

def get_input_oes_data(oes_file_list, thickness):
    X = pd.DataFrame()
    for oes_file in oes_file_list:
        unfolded_oes = data_unfolding.unfolding_oes(oes_file, thickness)
        X = pd.concat([X, unfolded_oes])
    output_path = f'./processed_data/{thickness}/OES_Data/X_OES_data_2.csv'
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

def normalize(X):
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median())
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


def __main__():
    thickness = '25um'
    fdc_folder_path = f'./TSV_Etch_Dataset/{thickness}/FDC_Data'
    #oes_folder_path = f'./TSV_Etch_Dataset/{thickness}/OES_Data'

    fdc_file_list = get_file_list(fdc_folder_path)
    #oes_file_list = get_file_list(oes_folder_path)

    #X = get_input_data(fdc_file_list, oes_file_list, thickness)
    X = get_input_fdc_data(fdc_file_list, thickness)
    #X = get_input_oes_data(oes_file_list, thickness)

    print(X)
    X = normalize(X)
    print(X)

    y = get_formatted_y(thickness)
    print(y)

    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)
    print("Random Forest Regressor has been created")


    loocv_rf = LOOCV(rf_regressor)
    print("LOOCV for RF_Regressor has been created")
    loocv_rf.fit(X, y)

    metrics = ['mse', 'mape', 'r2']
    for metric in metrics:
        print(f"LOOCV of Random Forest Regression using {metric.upper()} : {loocv_rf.mean_score(metric):.4f}")
    print(f"Random Forest Score for R Squared : {rf_regressor.score(X, y)[2]:.4f}")

if __name__ == '__main__':
    __main__()
