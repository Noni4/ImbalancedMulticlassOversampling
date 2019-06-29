from collections import Counter

import numpy as np

from instance_types import Types
from utils.uci_util import UciConnector
from algorithms_.type_classifier import StaticKNNTypeClassifier, DynamicKNNTypeClassifier
from utils.ml_util import dataframe_to_arrays
from algorithms_.preprocessing import Preprocessor

data = np.loadtxt('../data/results/localOptimizedDecisionTree', delimiter=';', dtype=object)

datasets = data[:,0]
datasets = set(datasets)

connector = UciConnector()
type_classifier = StaticKNNTypeClassifier()
preprocessor = Preprocessor()

counters_per_dataset = {}

for dataset in datasets:
    data, class_index = connector.get_dataset(dataset)
    X, y = dataframe_to_arrays(data, class_index)
    X = preprocessor.encode(X)
    counters_per_dataset[dataset] = Counter(type_classifier.get_types(X, y))

result_string = 'Dataset & Safe & Borderline & Rare & Outlier \\\\ \n'
for key, value in sorted(counters_per_dataset.items(), key=lambda item: item[0]):
    result_string += f"{key} & {value[Types.SAFE]} & {value[Types.BORDERLINE]} & {value[Types.RARE]} & {value[Types.OUTLIER]} \\\\ \n"

print(result_string)