

from sklearn.model_selection import train_test_split, cross_val_score

from algorithms_.oversampling import AllClassOversample, OneClassOversample, IndependentLocallyOptimizedAllClassOversample, DependetLocallyOptimizedAllClassOversample
from algorithms_.preprocessing import Preprocessor
from algorithms_.type_classifier import StaticKNNTypeClassifier, DynamicKNNTypeClassifier
from instance_types import Types
from utils.ml_util import dataframe_to_arrays, RandomState, run_cross_val, get_class_count, get_classifier
from utils.uci_util import UciConnector
from utils import logging_util
from utils.config_util import get_experiment_configuration, powerset
from utils.file_util import write_result
from utils.types_util import string_list_to_types_list

import numpy as np
import pandas as pd

logger = logging_util.get_new_logger('run_experiments')

dataset_names = get_experiment_configuration('dataset')
logger.debug(f"Dataset names: {dataset_names}")

oversampling_algorithms = get_experiment_configuration('oversamplingAlgorithm')
logger.debug(f"Oversampling Algorithms: {oversampling_algorithms}")

type_algorithms = get_experiment_configuration('typeAlgorithm')
logger.debug(f"Type Algorithms: {type_algorithms}")

algorithms = get_experiment_configuration('algorithm')
logger.debug(f"Algorithms: {algorithms}")

types = get_experiment_configuration('type')
logger.debug(f"Types: {types}")

seeds = get_experiment_configuration('seed')
logger.debug(f"Seeds: {seeds}")

classifier_names = get_experiment_configuration('classifier')
logger.debug(f"Classifier: {classifier_names}")

for classifier_name in classifier_names:
    for dataset_name in dataset_names:
        for type_algorithm in type_algorithms:
            for algorithm in algorithms:
                for oversampling_algorithm in oversampling_algorithms:
                    for type_subset in powerset(types):
                        for seed in seeds:
                            random_state = int(seed)
                            np.random.seed(random_state)
                            logger.debug(f"Currently working on: {dataset_name}, {type_algorithm}, {algorithm}, {type_subset}, {seed}")

                            uci_connector = UciConnector()
                            preprocessor = Preprocessor()

                            if type_algorithm == 'static':
                                type_classifier = StaticKNNTypeClassifier()
                            elif type_algorithm == 'dynamic':
                                type_classifier = DynamicKNNTypeClassifier()
                            else:
                                logger.error(f"Type algorithm {type_algorithm} is unknown")
                                continue

                            oversampler = None
                            if algorithm == 'oneClass':
                                oversampler = OneClassOversample()
                            elif algorithm == 'allClasses':
                                oversampler = AllClassOversample()
                            elif algorithm == 'independentLocallyOptimized':
                                oversampler = IndependentLocallyOptimizedAllClassOversample()
                            elif algorithm == 'dependedLocallyOptimized':
                                oversampler = DependetLocallyOptimizedAllClassOversample()
                            else:
                                logger.error(f"Number of classes to oversample {algorithm} unknown")

                            dataset, class_index = uci_connector.get_dataset(dataset_name)

                            X, y = dataframe_to_arrays(dataset, class_index)
                            X = preprocessor.encode(X)

                            instance_types = type_classifier.get_types(X, y)

                            types_list = string_list_to_types_list(type_subset)

                            if algorithm == 'independentLocallyOptimized':
                                X, y, oversampled_types, oversampled_classes = oversampler.oversample(X, y, instance_types, classifier_name, random_state, algorithm=oversampling_algorithm)
                            elif algorithm == 'biggestFirstDependentLocallyOptimized':
                                X, y, oversampled_types, oversampled_classes = oversampler.oversample(X, y, instance_types,
                                                                                                      classifier_name,
                                                                                                      random_state,
                                                                                                      first_class='biggest',
                                                                                                      algorithm=oversampling_algorithm)
                            elif algorithm == 'smallestFirstDependentOptimized':
                                X, y, oversampled_types, oversampled_classes = oversampler.oversample(X, y, instance_types,
                                                                                                      classifier_name,
                                                                                                      random_state,
                                                                                                      first_class='smallest',
                                                                                                      algorithm=oversampling_algorithm)
                            else:
                                X, y, oversampled_classes = oversampler.oversample(X, y, instance_types, np.array(types_list), random_state, algorithm=oversampling_algorithm)

                            if X is None:
                                logger.error(f"X is None, something went terribly wrong. Most likely the oversampling did not work")
                                continue

                            classifier = get_classifier(classifier_name, random_state)

                            if 'Optimized' in algorithm:
                                write_result(dataset_name,type_algorithm, algorithm, oversampled_types, seed,
                                             run_cross_val(classifier, X, y), oversampled_classes, classifier_name, oversampling_algorithm)
                            else:
                                write_result(dataset_name, type_algorithm, algorithm, type_subset, seed,
                                             run_cross_val(classifier, X, y), oversampled_classes, classifier_name, oversampling_algorithm)