import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .output_histogram import *
from .small_dist_reduction import *
from .label_encoding import *
from .interaction_regression import *

def get_outlier_out(y):
    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)
    y = get_filtered_y(y, outliers)

    filtering_idx = get_filtering_idx(outliers)

    return y, filtering_idx

def get_bin_reducible_vars(variance_df):
    q1_values = variance_df['variance'].quantile(0.25)
    below_q1 = variance_df[variance_df['variance'] <= q1_values]['index']
    below_q1 = below_q1.tolist()
    print(f'Number of Binary Variables that will be reduced : {len(below_q1)}')
    print(f'Reduction Binary Variable of {below_q1}')
    return below_q1

def get_bin_box_plot(variance_df):
    variance_df.boxplot()

    box_width = 0.15
    box_center = 1

    mean_variance = variance_df.mean(numeric_only=True).item()

    median_mean_variance = variance_df.median(numeric_only=True).item()

    #for i, row in variance_df.iterrows():
        # plt.hlines(y=row['variance'], xmin=box_center - box_width / 2, xmax=box_center + box_width / 2,
        #          color='red', linestyle='--', linewidth=1)
        #plt.text(x=box_center + box_width, y=row['variance'],
        #         s=f"{row['index']} ({row['variance']:.2f})",
        #         color='black', va='center', fontsize=10)

    plt.hlines(y=mean_variance, xmin=box_center - box_width / 2,
               xmax=box_center + box_width / 2, color='blue', linestyle='--', linewidth=1)
    plt.text(x=box_center - box_width, y=mean_variance,
             s=f"Mean ({mean_variance:.2f})",
             color="black", va='center', ha='right', fontsize=10)

    plt.text(x=box_center - box_width, y=median_mean_variance,
             s=f"Median ({median_mean_variance:.2f})",
             color="black", va='center', ha='right', fontsize=10)

    plt.title('Mean Variance of Binary Variables', fontsize=10)

    plt.show()

def free_bin_variable_reduction(X_data, viz=True):
    variance_df = get_binary_variance_df(X_data)

    if viz:
        get_bin_box_plot(variance_df)

    reducible_variable = get_bin_reducible_vars(variance_df)

    return reducible_variable

def get_small_var_out(X_data, y):
    print("get_small_var_out")
    reducible_vars = []
    cat_red_var = cat_variable_reduction(X_data, y)
    reducible_vars.extend(cat_red_var)
    ########################
    bin_red_var = free_bin_variable_reduction(X_data)
    reducible_vars.extend(bin_red_var)

    X_reduced = X_data.drop(columns=reducible_vars)
    print(f'Number of Stayed Variables : {X_reduced.shape[1]}')
    return X_reduced

def get_label_encoding(X_reduced):
    print("get_label_encoding")
    X_label_encoded = get_label_encoded_X(X_reduced)

    return X_label_encoded

def get_importance_features(X_train, y, threshold=0.95, n_estimators=50):
    print("get_importance_features")
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=3)
    rf.fit(X_train, y)

    importances = rf.feature_importances_
    feature_names = X_train.columns

    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = feature_names[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances from Random Forest')
    plt.show()

    cumulative_importance = np.cumsum(sorted_importances)
    optimal_num_features = np.argmax(cumulative_importance >= threshold) + 1
    print(f"{threshold * 100}% 설명력을 달성하는 최적 변수 개수: {optimal_num_features}")

    selected_features = sorted_features[:optimal_num_features]
    print(f"선택된 변수 목록: {selected_features}")

    return selected_features
    

def get_free_preprocessed_data():
    y = get_output()

    y, filtering_idx = get_outlier_out(y)

    X_data = get_X_data()

    X_reduced = get_small_var_out(X_data, y)

    X_labeled = get_label_encoding(X_reduced)

    X_filtered = get_filtered_X_data(X_labeled, filtering_idx)

    X_interaction_df = get_interaction_df(X_filtered)
    X_interaction_df.index = y.index

    selected_features = get_importance_features(X_interaction_df, y)

    X_test = pd.read_csv('./data/dataset/dataset/X_test.csv')
    X_test_label_encoded = get_label_encoded_X(X_test)
    test_poly = get_interaction_df(X_test_label_encoded)

    X_train_selected = X_interaction_df[selected_features]
    X_test_selected = test_poly[selected_features]


    return X_train_selected, X_test_selected, y