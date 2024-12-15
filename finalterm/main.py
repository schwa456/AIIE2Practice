import pandas as pd
from fianl_term import *

def get_datas():
    y = get_output()

    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)

    y = get_filtered_y(y, outliers)

    filtering_idx = get_filtering_idx(outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)
    X_reduced = get_reduced_X(filtered_X_data, y)
    X_label_encoded = get_label_encoded_X(X_reduced)

    interaction_df = get_interaction_df(X_label_encoded)

    interaction_df.index = y.index

    linear_model, X_poly = interaction_linear_regression(interaction_df, y)

    X_poly_removed = remove_over_p(X_label_encoded, X_poly, linear_model)

    X_poly_columns = X_poly_removed.columns
    print(X_poly_columns)

    X_test = pd.read_csv('./data/dataset/dataset/X_test.csv')
    X_test_label_encoded = get_label_encoded_X(X_test)
    test_interaction_df = get_interaction_df(X_test_label_encoded)
    test_interaction_df = sm.add_constant(test_interaction_df)


    test_interaction_df = test_interaction_df[X_poly_columns]
    print(test_interaction_df.columns)

    #test_linear_model, X_test_poly = interaction_linear_regression(test_interaction_df, y)

    #X_test_poly_removed = remove_over_p(X_test_label_encoded, X_test_poly, test_interaction_df)

    return X_poly_removed, test_interaction_df, y

# problem 1
def problem1(X, y, X_test, epsilon, threshold):
    X_projected = grp(X, epsilon=epsilon)

    score, model = pls(X_projected, y, threshold=threshold)

    y_pred = model.pls.predict(X_test)

    print(f"Prediction : {y_pred.flatten()}")
    print(f"Score : {score}")

    y_pred.to_csv('./1번.csv')
    return y_pred, score

def problem2(X, y, length, sigma, noise_sigma):
    pass

def __main__():
    X, X_test, y = get_datas()

    epsilon = 0.3
    threshold = 0.95
    problem1(X, y, epsilon=epsilon, threshold=threshold)

    """
    # TODO need to confirm hyperparameters
    length = 0
    sigma = 0
    noise_sigma = 0
    problem2(X, y, length, sigma, noise_sigma)
    """
if __name__ == '__main__':
    __main__()