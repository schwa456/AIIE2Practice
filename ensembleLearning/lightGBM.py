import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the dataset into LightGBM's Dataset format
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# Set the LightGBM parameters
params = {
    "objective": "regression",          # Regression problem
    "metric": "rmse",                   # root mean squared error
    "boosting_type": "gbdt",            # Gradient Boosting Decision Tree
    "num_leaves": 31,                   # Number of leaves in one tree
    "learning_rate": 0.1,               # Learning rate
    "feature_fraction": 0.8,            # Feature subsampling
    "bagging_fraction": 0.8,            # Data subsampling
    "bagging_freq": 5,                  # Frequency for bagging
    "seed": 42                          # Random seed for reproducibility
}

# Train the LightGBM model
num_boost_round = 100 # sNumber of boosting rounds
model = lgb.train(params, lgb_train, num_boost_round, valid_sets=[lgb_train, lgb_test])

# Make prediction
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"LightGBM Mean Squared Error: {mse:.2f}")