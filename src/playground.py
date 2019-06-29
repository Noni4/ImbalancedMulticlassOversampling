from scipy.stats import ttest_ind_from_stats

from analyzing import Analyzer
from utils.uci_util import UciConnector
from utils.ml_util import dataframe_to_arrays, split, get_majority_class, RandomState
from algorithms_.type_classifier import StaticKNNTypeClassifier, DynamicKNNTypeClassifier
from algorithms_.oversampling import OneClassOversample, AllClassOversample, \
    IndependentLocallyOptimizedAllClassOversample
from instance_types import Types
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import numpy as np
import pandas as pd

# uci_connector = UciConnector()
# analyzer = Analyzer()
# type_classifier = StaticKNNTypeClassifier()
# oversampler = AllClassOversample()
# random_state = RandomState()
#
# dataset, index = uci_connector.get_dataset('wine')
# X, y = dataframe_to_arrays(dataset, index)
#
# instance_types = type_classifier.get_types(X, y)
#
# X, y = oversampler.oversample(X, y, instance_types, [], algorithm='MCRBO')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state.random_state, test_size=0.25)
#
# classifier = DecisionTreeClassifier(random_state=random_state.random_state)
# print(str(np.mean(cross_val_score(classifier, X, y, cv=10))))

# classifier = DecisionTreeClassifier(random_state=get_random_state())
# classifier.fit(X_train, y_train)
#
# print(np.mean(cross_val_score(classifier, X_test, y_test, cv=10)))

# X = np.array([['Male', 1], ['Female', 3], ['Female', 2]])
# X = pd.DataFrame(X)
#
# print(f"Unencoded dataset: {X}")
#
# rows_to_encode = X.iloc[:,1:2]
# label_encoder = LabelEncoder()
# labeled_rows = rows_to_encode.apply(label_encoder.fit_transform)
#
# print(labeled_rows)
#
# one_hot_encoder = OneHotEncoder(categories='auto')
# encoded_rows = pd.DataFrame(one_hot_encoder.fit_transform(labeled_rows).toarray())
#
# print(encoded_rows)
# X = X.drop(X.iloc[:,1:2].columns, axis=1)
# X: pd.DataFrame = pd.concat([X, encoded_rows], axis=1, sort=False)
#
# print(f"Encoded datasset: {X}")

# t, p = ttest_ind_from_stats(89.99, 0.33, 20, 89.77, 0.31, 20)
# print(t,p)

X = np.array([[0.0],[1.0],[0.0],[0.5],[0.0],[1.0],[0.5],[0.3],[0.4]])
y = np.array([0,0,0,0,0,0,1,1,1])

type_classifier = StaticKNNTypeClassifier()
types = type_classifier.get_types(X, y)

over = IndependentLocallyOptimizedAllClassOversample()
over.oversample(X, y, types, 'SMOTE')