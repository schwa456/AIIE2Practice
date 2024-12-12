from fianl_term import *

def get_poly_removed_X():
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

    return X_poly_removed

def get_X_data_of_method(method, X_poly_removed):
    if method == 'normal':
        return X_poly_removed
    elif method == 'linear pca':
        pca = DynamicPCA()
        X_reduced = pca.fit_transform(X_poly_removed)
        X_linear_pca = pd.DataFrame(X_reduced)
        return X_linear_pca
    elif method == 'kernel pca':
        kpca = DynamicKernelPCA(kernel='rbf', gamma=15, target_variance=0.9)
        X_reduced = kpca.fit_transform(X_poly_removed)
        X_kernel_pca = pd.DataFrame(X_reduced)
        return X_kernel_pca
    elif method == 'grp':
        epsilon = 0.2
        X_reduced = grp(X_poly_removed, epsilon)
        X_grp = pd.DataFrame(X_reduced)
        return X_grp
    else:
        raise ValueError("'method' must be one of these : 'normal', 'linear pca', 'kernel pca', 'grp'")

def get_score(model, X_data, y):
    if model == 'random forest':
        score = random_forest(X_data, y)
    elif model == 'pls':
        score = pls(X_data, y)
    elif model == 'kernel ridge':
        score = kernel_ridge(X_data, y)
    elif model == 'gpr':
        score = gpr(X_data, y)
    else:
        raise ValueError

    return score

def __main__():
    methods = ['normal',
               #'linear pca',
               #'kernel pca',
               #'grp'
               ]
    models = ['random forest', 'pls', 'kernel ridge', 'gpr']

    y = get_output()

    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)

    y = get_filtered_y(y, outliers)

    result_dict = {}

    X_poly_removed = get_poly_removed_X()

    for method in methods:
        method_result = []
        for model in models:
            print("=" * 50)
            print(f"method : {method}, model : {model}")
            X_data = get_X_data_of_method(method, X_poly_removed)
            score = get_score(model, X_data, y)
            method_result.append(score)
            print("=" * 50)
        result_dict[method] = method_result

    result_df = pd.DataFrame(result_dict)

    result_df.to_csv('./check/comparison.csv')

if __name__ == '__main__':
    __main__()