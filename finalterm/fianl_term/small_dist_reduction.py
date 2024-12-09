import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from output_histogram import *

X_TRAIN_PATH = '../data/dataset/dataset/X_train.csv'

def get_X_data():
    file_path = X_TRAIN_PATH
    X_data = pd.read_csv(file_path)

    return X_data

def get_filtered_X_data(X_data, filtering_idx):
    filtered_X_data = X_data.drop(index=filtering_idx)
    return filtered_X_data

#######################################################

def get_cat_vars(X_data):
    cat_vars = X_data.select_dtypes(include='object').columns.tolist()
    return cat_vars

def get_cat_variance_df(X_data, y):
    cat_vars = get_cat_vars(X_data)
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
    return variance_df

def get_cat_reducible_vars(variance_df):
    q1_values = variance_df['mean_variance'].quantile(0.25)
    below_q1 = variance_df[variance_df['mean_variance'] <= q1_values]['index']
    below_q1 = below_q1.tolist()
    print(f'Number of Categorical Variables that will be reduced : {len(below_q1)}')
    print(f'Reduction Categorical Variable of {below_q1}')
    return below_q1


def get_cat_box_plot(variance_df):
    variance_df.boxplot()

    box_width = 0.15
    box_center = 1

    mean_mean_variance = variance_df.mean(numeric_only=True).item()

    median_mean_variance = variance_df.median(numeric_only=True).item()

    for i, row in variance_df.iterrows():
        plt.hlines(y=row['mean_variance'], xmin=box_center - box_width / 2, xmax=box_center + box_width / 2,
                  color='red', linestyle='--', linewidth=1)
        plt.text(x=box_center + box_width, y=row['mean_variance'],
                 s=f"{row['index']} ({row['mean_variance']:.2f})",
                 color='black', va='center', fontsize=10)

    plt.hlines(y=mean_mean_variance, xmin=box_center - box_width / 2,
               xmax=box_center + box_width / 2, color='blue', linestyle='--', linewidth=1)
    plt.text(x=box_center - box_width, y=mean_mean_variance,
             s=f"Mean ({mean_mean_variance:.2f})",
             color="black", va='center', ha='right', fontsize=10)

    plt.text(x=box_center - box_width, y=median_mean_variance,
             s=f"Median ({median_mean_variance:.2f})",
             color="black", va='center', ha='right', fontsize=10)

    plt.title('Mean Variance of Categorical Variables', fontsize=10)

    plt.show()


def cat_variable_reduction(X_data, y):
    variance_df = get_cat_variance_df(X_data, y)

    get_cat_box_plot(variance_df)

    reducible_variable = get_cat_reducible_vars(variance_df)

    return reducible_variable

################################################

def get_binary_vars(X_data):
    binary_vars = X_data.columns.tolist()[9:]
    return binary_vars

def get_binary_variance_df(X_data):
    bin_vars = get_binary_vars(X_data)
    result = {}
    for var in bin_vars:
        result[var] = X_data[var].var()
    variance_df = pd.DataFrame.from_dict(result, orient='index')
    variance_df = variance_df.rename(columns={0 : 'variance'})
    variance_df = variance_df.reset_index()
    return variance_df

def get_bin_scatter_plot(variance_df):
    variance_df = variance_df.copy()
    variance_df['index'] = variance_df['index'].apply(lambda x: x.replace('X', ''))

    variance_df.plot.scatter(x='index', y='variance', c=['red' if val==0 else 'blue' for val in variance_df['variance']])


    plt.xlabel('Binary Features')
    plt.ylabel('Variance')
    plt.title('Variance of Binary Variables')
    plt.xticks([])

    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

    plt.show()

def get_zero_variance(variance_df):
    zero_df = variance_df[variance_df['variance'] == 0]
    return zero_df

def get_duplicated_variance(variance_df):
    duplicated_df = variance_df[variance_df['variance'].duplicated(keep=False)]
    return duplicated_df


def bin_variable_reduction(X_data):
    variance_df = get_binary_variance_df(X_data)

    get_bin_scatter_plot(variance_df)

    zero_df = get_zero_variance(variance_df)
    zero_idx = zero_df['index'].tolist()

    duplicated_df = get_duplicated_variance(variance_df)
    duplicated_idx = duplicated_df['index'].tolist()

    red_idx = zero_idx + duplicated_idx
    red_idx = pd.Series(red_idx).drop_duplicates().tolist()

    print(f'Number of Binary Variables that will be Reduced : {len(red_idx)}')

    return red_idx

################################################

def get_reduced_X(X_data, y):
    reducible_vars = []
    cat_red_var = cat_variable_reduction(X_data, y)
    reducible_vars.extend(cat_red_var)
    ########################
    bin_red_var = bin_variable_reduction(X_data)
    reducible_vars.extend(bin_red_var)

    X_reduced = X_data.drop(columns=reducible_vars)
    print(f'Number of Stayed Variables : {X_reduced.shape[1]}')
    return X_reduced


################################################

def __main__():
    y = get_output()
    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)
    filtering_idx = get_filtering_idx(outliers)
    y = get_filtered_y(y, outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)


    X_reduced = get_reduced_X(filtered_X_data, y)


if __name__ == '__main__':
    __main__()