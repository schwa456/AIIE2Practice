import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class AdaBoostClassifier:
    def __init__(self, n_estimator=50):
        self.n_estimator = n_estimator
        self.alpha = [] # store the weight of each weak learner
        self.models = [] # store the weak learners

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Initializing weights uniformly
        weights = np.ones(n_samples)/n_samples

        for _ in range(self.n_estimator):
            # train a weak learner
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=weights)

            # make a prediction
            y_pred = stump.predict(X)

            # calculate the error rate
            error = np.sum(weights * (y_pred != y)) / np.sum(weights)

            # calculate the weight of the weak learner
            alpha = 0.5 * np.log((1-error) / (error + 1e-10))

            # Update Sample weights
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)

            # Save the model and its weight
            self.models.append(stump)
            self.alpha.append(alpha)

    def predict(self, X):
        # Weighted sum of prediction from all weak learners
        predictions = np.zeros(X.shape[0])

        for alpha, model in zip(self.alpha, self.models):
            predictions += alpha * model.predict(X)

        # Sign function to determine the final prediction
        return np.sign(predictions)

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# For AdaBoost, we need a binary classification, so we'll select only two class
X = X[y != 2]
y = y[y != 2]
y[y==0] = -1 # Convert to -1 and 1 for AdaBoost

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(n_estimator=50)
adaboost_clf.fit(x_train, y_train)

# Make predictions
y_pred = adaboost_clf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Classifier Accuracy: {accuracy:.2f}")