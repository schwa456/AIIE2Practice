import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed


class MRMR:
    def __init__(self):
        pass

    def get_mRMR(self, X, y, num_features):
        """
        Feature Selection using Mutual Information Regression

        Parameters:
        - X: 2-dimensional dataframe(# of sample x num of feature) which is input data matrix
        - y: 1-dimensional dataframe of Series which is target variable
        - num_features: number of features user wants to remain

        Returns:
        - selected_features: index list of selected features
        """

        # initializing the variables
        n_features = len(X.columns)
        selected_features = []
        remaining_features = list(X.columns)

        # calculate mutual information
        relevance = Parallel(n_jobs=-1)(delayed(mutual_info_regression)(X[[col]], y) for col in remaining_features)
        relevance = pd.Series([rel[0] for rel in relevance], index=remaining_features)

        # select first feature in terms of maximizing relevance
        first_feature = relevance.idxmax()
        selected_features.append(first_feature)
        remaining_features.remove(first_feature)

        # select features iteratively in remaining features
        for _ in range(1, num_features):
            mrmr_score = Parallel(n_jobs=-1)(
                delayed(lambda feature: relevance[feature] - calculate_redundancy(X, selected_features, feature))(feature)
                for feature in remaining_features)

            # select best mrmr score
            best_feature_idx = np.argmax(mrmr_score)
            best_feature = remaining_features[best_feature_idx]

            # 선택된 특징 update
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

        return selected_features

def calculate_redundancy(X, selected_features, candidate_feature):
    return np.mean([X[selected].corr(X[candidate_feature]) for selected in selected_features])