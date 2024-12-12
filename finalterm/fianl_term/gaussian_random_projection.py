import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

def calculate_n_components(X, epsilon):
    n_samples = X.shape[0]
    print(n_samples)
    numerator = 4 * np.log(n_samples)
    print(numerator)
    denominator = (epsilon**2 / 2)-(epsilon**3 / 3)
    print(denominator)

    result = int(np.ceil(numerator/denominator))
    print(result)
    return result

def grp(X, epsilon):
    n_components = calculate_n_components(X, epsilon)
    print(f"Number of Selected Components: {n_components}")

    grp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_projected = grp.fit_transform(X)

    print(f"Projected data shape: {X_projected.shape}")

    return X_projected

def __main__():
    X_poly_removed = pd.read_csv('./check/X_poly_removed.csv')

    epsilon = 0.2

    X_projected = grp(X_poly_removed, epsilon)
    print(X_projected.shape)
    grp_X = pd.DataFrame(X_projected)
    grp_X.to_csv('./check/grp_X.csv')


if __name__ == '__main__':
    __main__()