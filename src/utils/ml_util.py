import logging
import sys
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from utils import logging_util

logger = logging_util.get_new_logger('ml_util')


def dataframe_to_arrays(data: pd.DataFrame, class_index: int) -> (np.ndarray, np.ndarray):
    data_array = data.values
    if class_index == 0:
        X = data_array[:, 1:]
        y = data_array[:, 0]
    else:
        X = data_array[:, :class_index]
        y = data_array[:, class_index]
    try:
        return X, np.asarray(y, dtype=np.float64)
    except ValueError:
        return X, y

def split(X, y, seed=1, test_ratio=0.75) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    return train_test_split(X, y, test_size=test_ratio, random_state=seed)

def get_majority_class(y: np.ndarray):
    return Counter(y).most_common(1)[0][0]

def get_class_count(y: np.ndarray):
    return Counter(y)

def run_cross_val(classifier, X: np.ndarray, y: np.ndarray):
    assert classifier is not None
    assert X is not None
    assert y is not None
    return np.mean(cross_val_score(classifier, X, y, cv=10))

def get_classifier(classifier_name: str, random_state):
    if classifier_name == 'DecisionTree':
        classifier = DecisionTreeClassifier(random_state=random_state)
    elif classifier_name == 'MLP':
        classifier = MLPClassifier(random_state=random_state)
    elif classifier_name == 'KNN':
        classifier = KNeighborsClassifier()
    elif classifier_name == 'SVM':
        classifier = SVC(random_state=random_state)
    elif classifier_name == 'NaiveBayes':
        classifier = GaussianNB()
    else:
        logger.critical(f"Classifier name {classifier_name} unknown")
        sys.exit(1)
    return classifier

class RandomState:

    _random_state = 42

    def __init__(self):
        pass

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = int(new_random_state)