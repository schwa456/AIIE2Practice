import pandas as pd
import pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from fianl_term import *
from finalterm.fianl_term.free_preprocessing import get_free_preprocessed_data


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

    X_test = pd.read_csv('./data/dataset/dataset/X_test.csv')
    X_test_label_encoded = get_label_encoded_X(X_test)
    test_poly = get_interaction_df(X_test_label_encoded)
    test_poly['const'] = 1.0


    test_poly = test_poly[X_poly_columns]
    print(test_poly.columns)

    #test_linear_model, X_test_poly = interaction_linear_regression(test_interaction_df, y)

    #X_test_poly_removed = remove_over_p(X_test_label_encoded, X_test_poly, test_interaction_df)

    return X_poly_removed, test_poly, y

def get_main_grp(X_train, X_test, epsilon):
    n_components = calculate_n_components(X_train, epsilon)

    grp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_train_projected = grp.fit_transform(X_train)
    X_test_projected = grp.fit_transform(X_test)

    return X_train_projected, X_test_projected

# problem 1
def problem1(X, y, X_test, epsilon, threshold):

    X_train_projected, X_test_projected = get_main_grp(X, X_test, epsilon=epsilon)

    score, model = pls(X_train_projected, y, threshold=threshold)

    with open('pls_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('pls_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    y_pred = loaded_model.pls.predict(X_test_projected)

    print(f"Prediction : {y_pred.flatten()}")
    print(f"Score : {score}")

    y_pred_df = pd.DataFrame(y_pred)

    y_pred_df.to_csv('./1번.csv')
    return y_pred, score

def problem2_test(X_train, y):
    lengths = [0.1, 1.0, 10.0]
    sigmas = [0.1, 1.0, 10.0]
    noise_sigmas = [1e-10, 1.0]

    max_score = -10000000
    max_length = 0
    max_sigma = 0
    max_noise_sigma = 0

    for length in lengths:
        for sigma in sigmas:
            for noise_sigma in noise_sigmas:
                print("=" * 50)
                score = gpr(X_train, y)
                if score > max_score:
                    max_score = score
                    max_length = length
                    max_sigma = sigma
                    max_noise_sigma = noise_sigma
                print(f"Score with length {length}, sigma {sigma}, noise sigma {noise_sigma} : {score}")
                print("="*50)

    print(f"Max Hyper Parameters : length({max_length}), sigma({max_sigma}), noise sigma({max_noise_sigma})")
    print(f"Max R^2 Score : {max_score}")

def get_main_gpr(X_train, X_test, y, length, sigma, noise_sigma):

    kernel = C(sigma, (1e-3, 1e3)) * RBF(length, (1e-2, 1e2)) + WhiteKernel(noise_level=noise_sigma, noise_level_bounds=(1e-10, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    gpr.fit(X_train, y)

    y_pred, y_std = gpr.predict(X_test, return_std=True)

    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y, 'ko', label='Training Data')
    plt.plot(X_test, y_pred, 'b-', label='Predicted Mean')
    plt.fill_between(X_test.to_numpy()[:, 0].ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std, color='blue', alpha=0.2,
                     label='95% Confidence Interval')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()

    print(f"Optimized Kernel: {gpr.kernel_}")

    return y_pred

def problem2(X_train, X_test, y, length, sigma, noise_sigma):
    y_pred = get_main_gpr(X_train, X_test, y, length, sigma, noise_sigma)

    y_pred_df = pd.DataFrame(y_pred)

    y_pred_df.to_csv('./2번.csv')


def __main__():
    X, X_test, y = get_datas()

    epsilon = 0.3
    threshold = 0.95
    problem1(X, y, X_test, epsilon=epsilon, threshold=threshold)

    #############################################

    X_train, X_test, y = get_free_preprocessed_data()
    problem2(X_train, X_test, y, length=0.1, sigma=0.1, noise_sigma=1e-10)

if __name__ == '__main__':
    __main__()