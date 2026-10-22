import numpy as np
from typing import List, Tuple

class NaiveBayesCategorical:
    """
    Categorical Naive Bayes classifier.
    
    Attributes:
        categories_per_feature (List[int]): Number of categories for each feature.
        alpha (float): Additive Laplace smoothing parameter.
    """
    
    def __init__(self, categories_per_feature: List[int], alpha: float = 1.0) -> None:
        self.categories_per_feature = categories_per_feature
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        self.class_counts_: np.ndarray = np.array([])
        self.priors_: np.ndarray = np.array([])
        self.conditional_probs_: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesCategorical":
        """
        Fit Naive Bayes classifier according to X, y.
        
        Args:
            X (np.ndarray): Training vectors of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: The fitted model.
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Observations count per class
        self.class_counts_ = np.array([np.sum(y == c) for c in self.classes_])

        # Prior probabilities P(class)
        self.priors_ = self.class_counts_ / n_samples

        self.conditional_probs_ = []

        # Vectorized calculation of conditional probabilities
        for j in range(n_features):
            n_values = self.categories_per_feature[j]
            counts = np.zeros((len(self.classes_), n_values))

            for i, c in enumerate(self.classes_):
                # Mask for the current class
                class_mask = (y == c)
                # Count occurrences of each category value for this feature & class
                counts[i, :] = [np.sum(class_mask & (X[:, j] == v)) for v in range(n_values)]

            # Apply Laplace smoothing
            smoothed = (counts + self.alpha) / (
                self.class_counts_[:, np.newaxis] + self.alpha * n_values
            )
            self.conditional_probs_.append(smoothed)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        
        Args:
            X (np.ndarray): Test vectors of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted target values for X.
        """
        predictions = []

        for x in X:
            # The score is initially the log prior
            scores = np.log(self.priors_)

            # Add log conditional probabilities
            for j, value in enumerate(x):
                scores += np.log(self.conditional_probs_[j][:, value])

            # Class with the highest log probability
            predictions.append(self.classes_[np.argmax(scores)])

        return np.array(predictions)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: int = 3, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into random train and test subsets."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], test_idx


def main() -> None:
    # ---------------------------------------
    # 1. Dataset Initialization
    # ---------------------------------------
    # First column (Outlook): 0=sunny, 1=overcast, 2=rainy
    # Second column (Temperature): 0=moderate, 1=hot
    # Third column (Humidity): 0=normal, 1=high
    # Fourth column (Wind): 0=weak, 1=strong
    X = np.array([
        [0, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0],
        [2, 0, 0, 0], [2, 0, 1, 0], [2, 0, 1, 1],
        [1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0],
        [2, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0]
    ], dtype=int)

    # Target variable (PlayTennis?): 0=no, 1=yes
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=int)

    print(f"Dataset shape: X={X.shape}, y={y.shape}\n")

    # ----------------------------------------
    # 2. Split dataset
    # ----------------------------------------
    X_train, X_test, y_train, y_test, test_idx = train_test_split(
        X, y, test_size=3, random_state=42
    )
    
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set:     X={X_test.shape}, y={y_test.shape}")
    print(f"Test indices: {test_idx}\n")

    # ----------------------------------------
    # 3. Model Training & Evaluation
    # ----------------------------------------
    model = NaiveBayesCategorical(categories_per_feature=[3, 2, 2, 2], alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    print(f"Real classes:      {y_test}")
    print(f"Predicted classes: {y_pred}")

    accuracy = np.mean(y_pred == y_test)
    print(f"Comparison:        {y_pred == y_test}")
    print(f"Accuracy:          {accuracy:.2%}")


if __name__ == "__main__":
    main()
