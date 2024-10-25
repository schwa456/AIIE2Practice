import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the data
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the dataset into XGBoost's DMatrix format
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# Set the XGBoost parameters
params = {
    "objective": "reg:squarederror",    # Use squared error for regression
    "max_depth": 3,                     # Maximum depth of a tree
    "eta": 0.1,                         # Learning rate
    "subsample": 0.8,                   # Subsample ratio of the training instances
    "colsample_bytree": 0.8,            # Subsample ratio of columns when construction
    "seed": 42                          # Random seed for reproducibility
}

# Train the XGBoost model
num_boost_round = 100 # Number of boosting rounds
model = xgb.train(params, dtrain, num_boost_round)

# Make predictions
y_pred = model.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost Mean Squared Error: {mse:.2f}")