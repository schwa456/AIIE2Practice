import pandas as pd
from .output_histogram import *
from .small_dist_reduction import *
from sklearn.preprocessing import LabelEncoder

def get_label_encoded_X(X_data):
    label_encoder = LabelEncoder()

    cat_vars = get_cat_vars(X_data)
    for var in cat_vars:
        X_data[var] = label_encoder.fit_transform(X_data[var])
    return X_data

def __main__():
    y = get_output()

    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)

    y = get_filtered_y(y, outliers)

    filtering_idx = get_filtering_idx(outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)


    X_reduced = get_reduced_X(filtered_X_data, y)

    X_reduced.to_csv('../check/X_reduced.csv', index=False)

    X_label_encoded = get_label_encoded_X(X_reduced)
    print(X_label_encoded)

    X_label_encoded.to_csv('../check/X_label_encoded.csv', index=False)

if __name__ == '__main__':
    __main__()