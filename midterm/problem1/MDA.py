import numpy as np
import pandas as pd
from dataPreprocessing import FDCdata
import data_unfolding
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

from midterm.problem1.data_unfolding import grouped_fdc


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

    fdc_grouped = grouped_fdc(fdc_df)

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

def mean_decrease_accuracy(model, X, y, n_repeats=10):
    """
    Mean Decrease Accuracy model of Variable Importance Ranking in Random Forest
    :param model: learned model
    :param X: input data(n_samples, n_features)
    :param y: target_data(n_samples)
    :param n_repeats: number of times to repeat the model
    :return: Importance of each feature in Numpy Array
    """

    # 1. get base accuracy from X
    baseline_accuracy = mean_squared_error(y, model.predict(X))

    # 2. initializing the array of importance of each feature
    feature_importances = pd.Series(np.zeros(X.shape[1]), index=X.columns)

    # 3. Repeat bootstrapping and mean decrease
    for feature in X.columns:
        accuracies = []

        for _ in range(n_repeats):
            # copy data from X and permute of ith column
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])

            # get accuracy score of permuted data
            permuted_accuracy = mean_squared_error(y, model.predict(X_permuted))

            # store accuracy score of permuted data
            accuracies.append(permuted_accuracy)

        # calculate decreased mean accuracy
        mean_accuracy_drop = baseline_accuracy - np.mean(accuracies)
        feature_importances[feature] = mean_accuracy_drop

    return feature_importances

def get_top_n_features(X, feature_importances, n):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    top_n_features = importance_df.sort_values('Importance', ascending=False).head(n)

    return top_n_features

def visualizing(X, feature_importances):
    plt.bar(range(X.shape[1]), feature_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Decrease in Accuracy')
    plt.title('Mean Feature Importance')
    plt.show()

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

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    feature_importances = mean_decrease_accuracy(model, X, y)

    print("MDA Feature Importance")
    print(feature_importances)

    visualizing(X, feature_importances)

if __name__ == '__main__':
    __main__()