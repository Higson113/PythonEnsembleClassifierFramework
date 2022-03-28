import copy
from numpy.random import choice
import numpy as np


class ClassifierSampleWeightWrapper:

    def __init__(self, classifier):
        self.classifier = classifier
        self.fitted_classifier = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            # dont resample data
            X_resampled = X
            y_resampled = y
        else:
            # resample x and y
            draw = np.random.choice(len(X), len(X), True, np.asarray(sample_weight, dtype='float64'))
            X_resampled = np.take(X[:], draw, axis=0)
            y_resampled = np.take(y[:], draw, axis=0)

        temp_clf = copy.deepcopy(self.classifier)
        # fit
        self.fitted_classifier = temp_clf.fit(X_resampled, y_resampled)

    def predict(self, X):
        return np.array(self.fitted_classifier.predict(X))
