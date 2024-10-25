import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class BaggingClassifier:
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.sample = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap Sampling
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Training base learner
            estimator = self._clone_estimator()
            estimator.fit(X_sample, y_sample)

            # Save trained model and sample index
            self.estimators.append(estimator)
            self.sample.append(sample_indices)

    def predict(self, X):
        # collect all prediction from estimator
        prediction = np.array([estimator.predict(X) for estimator in self.estimators])

        # majority vote
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=prediction)
        return majority_vote

    def _clone_estimator(self):
        return DecisionTreeClassifier()

# load the example dataset
data = load_iris()
X, y = data.data, data.target

# split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing Bagging Classifier and Learning
bagging_clf = BaggingClassifier(n_estimators=10)
bagging_clf.fit(X_train, y_train)

# predict
y_pred = bagging_clf.predict(X_test)

# assess the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Bagging Classifier: {accuracy:.2f}")
