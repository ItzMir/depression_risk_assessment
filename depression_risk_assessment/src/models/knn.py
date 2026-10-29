"""
Handwritten K-Nearest Neighbors classifier.

Distance  : Euclidean
Scaling   : features standardized (mean 0, std 1) using training data only
Prediction: majority class among the k nearest training points
Probability: fraction of class-1 among the k neighbors
"""

import math


class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []
        self.means = []
        self.stds = []

    # ---------------- standardization ----------------
    def _fit_scaler(self, X):
        n = len(X)
        d = len(X[0]) if n > 0 else 0
        self.means = [0.0] * d
        self.stds = [1.0] * d
        for j in range(d):
            col = [row[j] for row in X]
            m = sum(col) / n
            var = sum((v - m) ** 2 for v in col) / n
            s = math.sqrt(var) if var > 0 else 1.0
            self.means[j] = m
            self.stds[j] = s

    def _transform(self, X):
        out = []
        for row in X:
            out.append([
                (row[j] - self.means[j]) / self.stds[j]
                for j in range(len(row))
            ])
        return out

    # ---------------- training ----------------
    def fit(self, X, y):
        self._fit_scaler(X)
        self.X_train = self._transform(X)
        self.y_train = list(y)
        return self

    # ---------------- prediction ----------------
    @staticmethod
    def _distance(a, b):
        s = 0.0
        for i in range(len(a)):
            d = a[i] - b[i]
            s += d * d
        return math.sqrt(s)

    def _neighbors(self, row):
        dists = []
        for i, train_row in enumerate(self.X_train):
            dists.append((self._distance(row, train_row), self.y_train[i]))
        dists.sort(key=lambda x: x[0])
        return dists[:self.k]

    def predict_proba(self, X):
        Xs = self._transform(X)
        out = []
        for row in Xs:
            neighbors = self._neighbors(row)
            p1 = sum(lbl for _, lbl in neighbors) / len(neighbors)
            out.append(p1)
        return out

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]