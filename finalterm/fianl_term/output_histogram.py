import pandas as pd
import matplotlib.pyplot as plt

FILTER_IDX = None
Y_PATH = '../data/dataset/dataset/y_train.csv'
QUANT_VAL = 0.9999

def get_output():
    file_path = Y_PATH
    y = pd.read_csv(file_path)

    return y

def get_histogram(data, outlier_boundary):
    data.plot.hist(bins=100, alpha=0.7, edgecolor='black')

    plt.title('Output Variable (y)')
    plt.xlabel('Test Times')
    plt.ylabel('Number of Data Points')

    plt.axvline(outlier_boundary, linestyle='--', color='red', linewidth=2, label='Outlier Boundary')
    plt.text(outlier_boundary, plt.gca().get_ylim()[1] * 0.9, f'outlier boundary: {outlier_boundary :.2f}  ',
             color='black', rotation=0, ha='right', va='center', fontsize=10)

    plt.show()

def get_outlier(data):
    outlier = data.quantile(QUANT_VAL).item()
    print(f"{QUANT_VAL * 100} percentile 값: {outlier :.2f}")
    return outlier

def get_number_of_outliers(y, outlier_boundary):
    outliers = y[y >= outlier_boundary]
    outliers = outliers.dropna()
    print(f'Number of Outliers: {len(outliers)}')
    print(f'Outlier Index: {outliers['y'].index.item()}')
    print(f'Outlier Value: {outliers['y'].item()}')
    return outliers

def get_filtered_y(y, outliers):
    filtered_y = y[~y.isin(outliers['y'])]
    filtered_y = filtered_y.dropna()
    return filtered_y

def get_filtering_idx(outliers):
    FILTER_IDX = outliers.index.item()
    return FILTER_IDX

def __main__():
    y = get_output()
    outlier_boundary = get_outlier(y)
    get_histogram(y, outlier_boundary)
    outliers = get_number_of_outliers(y, outlier_boundary)
    filtered_y = get_filtered_y(y, outliers)


if __name__ == '__main__':
    __main__()