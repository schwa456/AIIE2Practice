import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from .output_histogram import *
from .small_dist_reduction import *
from .label_encoding import *

def get_interaction_df(X_data):
    poly = PolynomialFeatures(degree = 2, interaction_only=True, include_bias=False)

    interaction_terms = poly.fit_transform(X_data)

    interaction_columns = poly.get_feature_names_out(X_data.columns)

    interaction_df = pd.DataFrame(interaction_terms, columns = interaction_columns)

    #interaction_df = interaction_df.loc[:, interaction_df.columns.str.contains(r'\s')]

    return interaction_df

def interaction_linear_regression(interaction_df, y):
    X_poly = sm.add_constant(interaction_df)
    model = sm.OLS(y, X_poly).fit()

    with open('./check/OLS_summary.txt', 'w') as f:
        f.write(model.summary().as_text())

    result = pd.DataFrame({
        'Coefficient': model.params,
        'Std Error': model.bse,
        't Value': model.tvalues,
        'P-Value': model.pvalues,
        'Confidence Interval(Lower)': model.conf_int()[0],
        'Confidence Interval(Upper)': model.conf_int()[1],
    })

    result.to_csv(f"./check/OLS_results.csv", index=False)

    return model, X_poly

def remove_over_p(X_data, X_poly, model):
    significant_columns = model.pvalues[model.pvalues <= 0.05].index

    original_columns = ['const'] + [col for col in X_data.columns if col in significant_columns]

    final_X = X_poly[significant_columns.union(original_columns)]

    return final_X

def __main__():
    y = get_output()

    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)

    y = get_filtered_y(y, outliers)
    y.to_csv('./check/y.csv')

    filtering_idx = get_filtering_idx(outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)
    X_reduced = get_reduced_X(filtered_X_data, y)
    X_label_encoded = get_label_encoded_X(X_reduced)

    interaction_df = get_interaction_df(X_label_encoded)
    print(interaction_df.shape)
    print(y.shape)

    interaction_df.index = y.index

    linear_model, X_poly = interaction_linear_regression(interaction_df, y)

    X_poly_removed = remove_over_p(X_label_encoded, X_poly, linear_model)
    X_poly_removed.to_csv('./check/X_poly_removed.csv')
    print(X_poly_removed.shape)
    print(X_poly_removed.columns)

if __name__ == '__main__':
    __main__()