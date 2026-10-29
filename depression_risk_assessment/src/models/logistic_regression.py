"""
Handwritten Logistic Regression with:
    - Sigmoid activation
    - Binary cross-entropy loss
    - L2 regularization
    - Gradient descent
    - Optional class weighting (default OFF; folds are already balanced)
    - Fold-local feature standardization (StandardScaler baked in)
"""

import math


# ---------------------------------------------------------------------------
# StandardScaler (formerly model_utils.py)
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Compute mean and std from training data only.
    Apply the same transform to any subsequent data.
    """

    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, X):
        n = len(X)
        d = len(X[0]) if n > 0 else 0
        self.means = [0.0] * d
        self.stds = [1.0] * d

        for j in range(d):
            col = [row[j] for row in X]
            m = sum(col) / n if n > 0 else 0.0
            var = sum((v - m) ** 2 for v in col) / n if n > 0 else 1.0
            s = math.sqrt(var) if var > 0 else 1.0
            self.means[j] = m
            self.stds[j] = s

    def transform(self, X):
        out = []
        for row in X:
            new_row = [
                (row[j] - self.means[j]) / self.stds[j]
                for j in range(len(row))
            ]
            out.append(new_row)
        return out

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def sigmoid(z):
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_lambda=0.01,
                 use_class_weights=False):
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.use_class_weights = use_class_weights
        self.weights = []
        self.bias = 0.0
        self.scaler = StandardScaler()
        self.class_weight = {0: 1.0, 1: 1.0}

    def _standardize_fit(self, X):
        return self.scaler.fit_transform(X)

    def _standardize(self, X):
        return self.scaler.transform(X)

    def _compute_class_weights(self, y):
        n = len(y)
        n_0 = sum(1 for v in y if v == 0)
        n_1 = sum(1 for v in y if v == 1)
        w0 = n / (2 * n_0) if n_0 > 0 else 1.0
        w1 = n / (2 * n_1) if n_1 > 0 else 1.0
        return {0: w0, 1: w1}

    def fit(self, X, y):
        Xs = self._standardize_fit(X)
        n = len(Xs)
        d = len(Xs[0]) if n > 0 else 0

        if self.use_class_weights:
            self.class_weight = self._compute_class_weights(y)
        else:
            self.class_weight = {0: 1.0, 1: 1.0}

        self.weights = [0.0] * d
        self.bias = 0.0

        for epoch in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0

            for i in range(n):
                z = self.bias + sum(self.weights[j] * Xs[i][j] for j in range(d))
                p = sigmoid(z)
                w_i = self.class_weight[y[i]]
                err = (p - y[i]) * w_i

                for j in range(d):
                    grad_w[j] += err * Xs[i][j]
                grad_b += err

            for j in range(d):
                grad_w[j] = (grad_w[j] / n) + self.l2_lambda * self.weights[j]
            grad_b = grad_b / n

            for j in range(d):
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        Xs = self._standardize(X)
        d = len(self.weights)
        out = []
        for row in Xs:
            z = self.bias + sum(self.weights[j] * row[j] for j in range(d))
            out.append(sigmoid(z))
        return out

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]