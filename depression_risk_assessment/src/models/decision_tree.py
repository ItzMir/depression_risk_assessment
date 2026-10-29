"""
Handwritten Decision Tree classifier.

Split criterion : Gini impurity
Threshold search: up to n_thresholds candidate values per feature,
                  drawn from the sorted unique values.
Stopping rules  : max_depth, min_samples_split, min_samples_leaf.
predict_proba   : fraction of class-1 samples in the reached leaf.
"""


def _gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    p1 = sum(y) / n
    p0 = 1.0 - p1
    return 1.0 - p0 * p0 - p1 * p1


def _candidate_thresholds(values, n_thresholds):
    uniq = sorted(set(values))
    if len(uniq) <= 1:
        return []
    step = max(1, len(uniq) // (n_thresholds + 1))
    return uniq[step::step][:n_thresholds]


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10,
                 min_samples_leaf=5, n_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        p1 = sum(y) / n if n > 0 else 0.0

        if (depth >= self.max_depth
                or n < self.min_samples_split
                or p1 == 0.0
                or p1 == 1.0):
            return {"leaf": True, "p1": p1}

        best = self._best_split(X, y)
        if best is None:
            return {"leaf": True, "p1": p1}

        feature, threshold = best
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n):
            if X[i][feature] <= threshold:
                left_X.append(X[i]); left_y.append(y[i])
            else:
                right_X.append(X[i]); right_y.append(y[i])

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(left_X, left_y, depth + 1),
            "right": self._build(right_X, right_y, depth + 1),
        }

    def _best_split(self, X, y):
        n = len(y)
        parent_gini = _gini(y)
        best_gain = 0.0
        best = None

        for j in range(len(X[0])):
            col = [row[j] for row in X]
            for t in _candidate_thresholds(col, self.n_thresholds):
                left_y, right_y = [], []
                for i in range(n):
                    if col[i] <= t:
                        left_y.append(y[i])
                    else:
                        right_y.append(y[i])
                if (len(left_y) < self.min_samples_leaf
                        or len(right_y) < self.min_samples_leaf):
                    continue
                wl = len(left_y) / n
                wr = len(right_y) / n
                gini = wl * _gini(left_y) + wr * _gini(right_y)
                gain = parent_gini - gini
                if gain > best_gain:
                    best_gain = gain
                    best = (j, t)
        return best

    def _leaf_p1(self, row, node):
        while not node["leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["p1"]

    def predict_proba(self, X):
        return [self._leaf_p1(row, self.root) for row in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]