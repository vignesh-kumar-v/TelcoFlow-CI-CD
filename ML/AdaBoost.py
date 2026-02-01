import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = None
        self.alpha = None
    def predict(self, X):
        num_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(num_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, num_classifiers=5):
        self.num_classifiers = num_classifiers
        self.classifiers = []
    def fit(self, X, y):
        num_samples, num_features = X.shape
        print(f"Num of Samples: {num_samples}")
        print(f"Num of features: {num_features}")
        w = np.full(num_samples, (1 / num_samples))
        for _ in range(self.num_classifiers):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(num_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(num_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1
                        error = sum(w[y != predictions])
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.classifiers.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.classifiers]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)                