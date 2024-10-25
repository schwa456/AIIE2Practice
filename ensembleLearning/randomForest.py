import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class RandomForestClassifierCustom:
    def __init__(self, n_estimators=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.samples = []

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Train each tree in the forest
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Select a random subset of features
            max_features = self._get_max_features(n_features)
            feature_indices = np.random.choice(n_features, max_features, replace=False)

            # Create and train the decision tree
            tree = DecisionTreeClassifier(random_state=None)
            tree.fit(X_sample[:, feature_indices], y_sample)

            # Store the trained tree and feature indices
            self.trees.append((tree, feature_indices))
            self.samples.append(sample_indices)

    def predict(self, X):
        predictions = []

        # Collect predictions from each tree
        for tree, feature_indices in self.trees:
            predictions.append(tree.predict(X[:, feature_indices]))

        # Perform majority voting for each sample
        predictions = np.array(predictions).T
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

        return majority_vote


# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifierCustom
rf_clf = RandomForestClassifierCustom(n_estimators=10, max_features='sqrt')
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")
