import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

class GBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators # The number of weak learner
        self.learning_rate = learning_rate # Step size
        self.max_depth = max_depth # The maximum depth of each weak learner
        self.models = [] # List to store weak learners
        self.initial_prediction = None # Initial prediction

    def fit(self, X, y):
        # Initialize with the mean of the target variable
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - y_pred

            # Train a decision tree on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

            # Store the trained tree
            self.models.append(tree)

    def predict(self, X):
        # Start with the initial prediction
        y_pred = np.full(X.shape[0], self.initial_prediction)

        # Add predictions from each weak learner
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

# Load a sample dataset (Boston Housing)
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the GBMCustom model
gbm = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"GBM Mean Squared Error with California Housing Data: {mse:.2f}")