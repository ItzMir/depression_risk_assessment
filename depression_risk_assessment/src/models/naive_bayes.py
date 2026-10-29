"""
Handwritten Gaussian Naive Bayes classifier.

Model:
    For each class c:
        - prior P(c) = count(c) / N
        - for each feature j: mean and variance of feature j in class c
    Likelihood:
        P(x_j | c) = Gaussian(x_j; mean_cj, var_cj + smoothing)
    Prediction:
        argmax_c  log P(c) + sum_j log P(x_j | c)
"""

import math


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = []
        self.priors = {}
        self.means = {}    # means[c] = list of per-feature means
        self.vars = {}     # vars[c]  = list of per-feature variances

    def fit(self, X, y):
        n = len(X)
        d = len(X[0]) if n > 0 else 0
        self.classes = sorted(set(y))

        for c in self.classes:
            rows = [X[i] for i in range(n) if y[i] == c]
            nc = len(rows)
            self.priors[c] = nc / n if n > 0 else 0.0

            means_c = [0.0] * d
            vars_c = [0.0] * d
            for j in range(d):
                col = [row[j] for row in rows]
                m = sum(col) / nc if nc > 0 else 0.0
                var = sum((v - m) ** 2 for v in col) / nc if nc > 0 else 1.0
                means_c[j] = m
                vars_c[j] = var
            self.means[c] = means_c
            self.vars[c] = vars_c

        # Global variance for smoothing
        self.global_var = 0.0
        for j in range(d):
            col = [row[j] for row in X]
            m = sum(col) / n if n > 0 else 0.0
            self.global_var += sum((v - m) ** 2 for v in col) / n if n > 0 else 0.0
        if d > 0:
            self.global_var /= d

        return self

    @staticmethod
    def _log_gaussian(x, mean, var):
        # log of N(x; mean, var)
        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    def _log_posterior(self, row, c):
        log_p = math.log(self.priors[c]) if self.priors[c] > 0 else -1e18
        smooth = self.var_smoothing * (self.global_var if self.global_var > 0 else 1.0)
        for j, x in enumerate(row):
            var = self.vars[c][j] + smooth
            if var <= 0:
                var = 1e-9
            log_p += self._log_gaussian(x, self.means[c][j], var)
        return log_p

    def predict_proba(self, X):
        out = []
        for row in X:
            logs = [self._log_posterior(row, c) for c in self.classes]
            # Softmax over the two log-posteriors
            m = max(logs)
            exps = [math.exp(l - m) for l in logs]
            total = sum(exps)
            # Return probability of class 1 (second element if classes == [0,1])
            if self.classes == [0, 1]:
                out.append(exps[1] / total if total > 0 else 0.0)
            else:
                # fallback: return the probability of the last class
                out.append(exps[-1] / total if total > 0 else 0.0)
        return out

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]