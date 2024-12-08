import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .output_histogram import *

X_TRAIN_PATH = '../data/dataset/dataset/X_train.csv'

def get_X_data():
    file_path = X_TRAIN_PATH
    X_data = pd.read_csv(file_path)

    return X_data

def get_filtered_X_data(X_data, filtering_idx):
    filtered_X_data = X_data.drop(index=filtering_idx)
    return filtered_X_data
"""
def get_cat_with_y_df(X_data, y):
    cat_vars = X_data.columns.tolist()[1:9]
    combined_datas = []
    for cat in cat_vars:
        combined = pd.merge(X_data[cat], y, left_index=True, right_index=True)
        combined_datas.append(combined)
        print(combined)
    return combined_datas
"""

def get_variance_df(X_data, y):
    cat_vars = X_data.columns.tolist()[1:9]
    combined_datas = []
    variance_for_var = {}
    for var in cat_vars:
        merged = pd.merge(X_data[var], y, left_index=True, right_index=True)
        combined = merged.groupby(var, as_index=False).agg({'y': 'mean'})
        combined_datas.append(combined)
        variance_for_var[var] = merged.groupby(var)['y'].var().mean().item()
    variance_df = pd.DataFrame.from_dict(variance_for_var, orient='index')
    variance_df = variance_df.rename(columns={0 : 'mean_variance'})
    variance_df = variance_df.reset_index()
    print(variance_df)
    return variance_df


def get_cat_box_plot(variance_df):
    variance_df.boxplot()

"""
    positions = range(1, len(variance_df['mean_variance'].unique()) + 1)

    for pos, var in zip(positions, variance_df['mean_variance'].unique()):
        plt.text(pos, variance_df['mean_variance'].max() * 1.05, var, ha='center', va='center', fontsize=10, fontweight='bold')

    plt.show()
"""

def cat_var_reduction(X_data, y):
    cat_vars = X_data.columns.tolist()[1:9]

    combined_data = pd.merge(X_data[cat_vars], y, left_index=True, right_index=True)


def get_binary_vars(X_data):
    binary_vars = X_data.columns.tolist()[10:]
    return binary_vars

def __main__():
    y = get_output()
    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)
    filtering_idx = get_filtering_idx(outliers)
    y = get_filtered_y(y, outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)

    variance_df = get_variance_df(filtered_X_data, y)

    get_cat_box_plot(variance_df)

    cat_var_reduction(filtered_X_data, y)

if __name__ == '__main__':
    __main__()