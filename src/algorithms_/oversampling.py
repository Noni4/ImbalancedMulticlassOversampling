from typing import Dict, List

from imblearn.over_sampling import SMOTE

from algorithms_.type_classifier import StaticKNNTypeClassifier

import numpy as np

from random import randint, seed

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from instance_types import Types
from utils.config_util import powerset
from utils.ml_util import get_majority_class, get_class_count, RandomState, run_cross_val, get_classifier

from utils import logging_util

import sys
from os import path

sys.path.append(path.abspath('../../MultiClassRBO'))

try:
    from algorithms import MultiClassRBO
except ModuleNotFoundError as e:
    print(f"MultiClassRBO is not installed, please install it in ../../MultiClasssRBO")

class Oversample:

    logger = logging_util.get_new_logger(__name__)

    def _one_class_oversample(self, X: np.ndarray, y: np.ndarray, instance_types: np.ndarray, algorithm: str,
                              all_classes: List, class_number_to_oversample: int, instance_types_to_oversample,
                              majority_class_name: str, random_state):
        self.logger.debug(f"Class counter before: {get_class_count(y)}")
        self.logger.debug(f"Class to oversample: {all_classes[class_number_to_oversample]}")
        self.logger.debug(f"Types to oversample: {instance_types_to_oversample}")

        minority_instance_number_not_to_oversample = []
        for i in range(len(y)):
            if y[i] == all_classes[class_number_to_oversample] and instance_types[
                i] not in instance_types_to_oversample:
                minority_instance_number_not_to_oversample.append(i)

        X_not_to_oversample = np.take(X, minority_instance_number_not_to_oversample, 0)
        y_not_to_oversample = np.take(y, minority_instance_number_not_to_oversample, 0)

        self.logger.debug(f"X not to oversample shape: {X_not_to_oversample.shape}")
        self.logger.debug(f"y not to oversample shape: {y_not_to_oversample.shape}")

        X_to_oversample = np.delete(X, minority_instance_number_not_to_oversample, 0)
        y_to_oversample = np.delete(y, minority_instance_number_not_to_oversample, 0)
        number_of_instances_for_oversampling = y_to_oversample.shape[0]

        self.logger.debug(f"X to oversample shape: {X_to_oversample.shape}")
        self.logger.debug(f"y to oversample shape: {y_to_oversample.shape}")
        self.logger.debug(f"Number of instances selected for oversampling: {number_of_instances_for_oversampling}")

        class_counter = get_class_count(y)
        number_of_initial_instances = class_counter[all_classes[class_number_to_oversample]]

        class_counter[all_classes[class_number_to_oversample]] = class_counter[majority_class_name] - len(
            minority_instance_number_not_to_oversample)

        if algorithm == 'SMOTE':
            try:
                smote = SMOTE(sampling_strategy=class_counter, random_state=random_state)
                X_oversampled, y_oversampled = smote.fit_resample(X_to_oversample, y_to_oversample)
            except ValueError as e:
                self.logger.warning(e)
                self.logger.warning(f"There are not enough samples of one class smote neighbour are reduced to 3")
                try:
                    smote = SMOTE(sampling_strategy=class_counter, random_state=random_state,
                                  k_neighbors=3)
                    X_oversampled, y_oversampled = smote.fit_resample(X_to_oversample, y_to_oversample)
                except ValueError as e:
                    self.logger.warning(e)
                    self.logger.warning(f"There are not enough samples of one class smote neighbour are reduced to 1")
                    try:
                        smote = SMOTE(sampling_strategy=class_counter, random_state=random_state,
                                      k_neighbors=1)
                        X_oversampled, y_oversampled = smote.fit_resample(X_to_oversample, y_to_oversample)
                    except ValueError as e:
                        self.logger.warning(e)
                        self.logger.warning(f"There are not enough samples of one class smote cannot be executed")
                        return X, y, 0
        elif algorithm == 'MCRBO':
            mcrbo = MultiClassRBO()
            X_oversampled, y_oversampled = mcrbo.fit_sample(X, y)
        else:
            self.logger.critical(f"Algorithm: {algorithm} unknown")
            sys.exit(1)

        X = np.vstack((X, X_oversampled[number_of_instances_for_oversampling:]))
        y = np.hstack((y, y_oversampled[number_of_instances_for_oversampling:]))

        class_counter = get_class_count(y)

        self.logger.debug(f"Class counter: {class_counter}")

        self.logger.debug(f"New X shape: {X.shape}")
        self.logger.debug(f"New y shape: {y.shape}")

        return X, y, get_class_count(y)[all_classes[class_number_to_oversample]] - number_of_initial_instances

class OneClassOversample(Oversample):

    def __init__(self):
        pass

    def oversample(self, X: np.ndarray, y: np.ndarray, instance_types: np.ndarray, instance_types_to_oversample: np.ndarray, random_state,
                   algorithm='SMOTE'):

        self.logger.debug(f"Old X shape: {X.shape}")
        self.logger.debug(f"Old y shape: {y.shape}")

        majority_class_name = get_majority_class(y)
        all_classes =  list(set(y.flatten()))
        class_number_to_oversample = randint(0, len(all_classes) - 1)
        while all_classes[class_number_to_oversample] == majority_class_name:
            class_number_to_oversample = randint(0, len(all_classes) - 1)

        X, y, _ = self._one_class_oversample(X, y, instance_types, algorithm, all_classes, class_number_to_oversample,
                                          instance_types_to_oversample, majority_class_name, random_state)
        return X, y


class AllClassOversample(Oversample):

    def __init__(self):
        pass

    def oversample(self, X: np.ndarray, y: np.ndarray, instance_types: np.ndarray, instance_types_to_oversample: np.ndarray, random_state,
                   algorithm='SMOTE'):

        self.logger.debug(f"Old X shape: {X.shape}")
        self.logger.debug(f"Old y shape: {y.shape}")

        majority_class_name = get_majority_class(y)
        all_classes =  list(set(y.flatten()))
        all_classes.remove(majority_class_name)

        oversampled_classes = []
        for class_number_to_oversample in range(len(all_classes)):
            X, y, oversampled = self._one_class_oversample(X, y, instance_types, algorithm, all_classes,
                                                           class_number_to_oversample, instance_types_to_oversample,
                                                           majority_class_name, random_state)
            oversampled_classes.append(oversampled)

        self.logger.debug(f"Final class count: {get_class_count(y)}")

        return X, y, oversampled_classes


class IndependentLocallyOptimizedAllClassOversample(Oversample):

    def __init__(self):
        pass

    def oversample(self, X: np.ndarray, y: np.ndarray, instance_types: np.ndarray, classifier_name, random_state, algorithm='SMOTE'):
        self.logger.debug(f"Initial class counter: {get_class_count(y)}")

        all_types_to_oversample = np.array([Types.SAFE, Types.BORDERLINE, Types.RARE, Types.OUTLIER])

        majority_class_name = get_majority_class(y)
        all_classes = list(set(y.flatten()))
        all_classes.remove(majority_class_name)

        all_subclasses = list(powerset(all_types_to_oversample))

        instance_types_per_class = []
        for class_number_to_oversample in range(len(all_classes)):
            scores = np.array([])
            for subclass in all_subclasses:
                self.logger.debug(f"Oversample: {all_classes[class_number_to_oversample]}, {subclass}")

                X_oversampled, y_oversampled, _ = self._one_class_oversample(X, y, instance_types, algorithm, all_classes,
                                                                    class_number_to_oversample, subclass,
                                                                    majority_class_name, random_state)

                scores = np.append(scores, run_cross_val(get_classifier(classifier_name, random_state), X_oversampled, y_oversampled))

            self.logger.debug(f"Scores: {scores}")

            highest_scores = np.argwhere(scores == np.amax(scores))
            if len(highest_scores) == 1:
                instance_types_per_class.append(highest_scores[0][0])
            elif 0 in highest_scores:
                instance_types_per_class.append(0)
            else:
                instance_types_per_class.append(highest_scores[np.random.randint(0, len(highest_scores))][0])

        oversampled_classes = []
        for class_number_to_oversample in range(len(all_classes)):
            X, y, oversampled = self._one_class_oversample(X, y, instance_types, algorithm, all_classes, class_number_to_oversample,
                                              all_subclasses[instance_types_per_class[class_number_to_oversample]], majority_class_name, random_state)
            oversampled_classes.append(oversampled)

        self.logger.debug(f"Instance types per class: {instance_types_per_class}")
        self.logger.debug(f"Final class count: {get_class_count(y)}")
        self.logger.debug(f"Oversampled classes: {oversampled_classes}")

        return X, y, [all_subclasses[type_number] for type_number in instance_types_per_class], oversampled_classes


class DependetLocallyOptimizedAllClassOversample(Oversample):

    def __init__(self):
        pass

    def oversample(self, X: np.ndarray, y: np.ndarray, instance_types: np.ndarray, classifier_name, random_state, first_class = 'biggest', algorithm='SMOTE'):
        self.logger.debug(f"Initial class counter: {get_class_count(y)}")

        all_types_to_oversample = np.array([Types.SAFE, Types.BORDERLINE, Types.RARE, Types.OUTLIER])

        majority_class_name = get_majority_class(y)
        all_classes_dict = get_class_count(y)
        all_classes_dict.pop(majority_class_name)

        if first_class == 'biggest':
            all_classes = [class_ for class_, occurence in sorted(all_classes_dict.items(), key=lambda kv: kv[1], reverse=True)]
        else:
            all_classes = [class_ for class_, occurence in sorted(all_classes_dict.items(), key=lambda kv: kv[1], reverse=False)]
        all_subclasses = list(powerset(all_types_to_oversample))

        instance_types_per_class = []
        oversampled_classes = []
        for class_number_to_oversample in range(len(all_classes)):
            scores = np.array([])
            for subclass in all_subclasses:
                self.logger.debug(f"Oversample: {all_classes[class_number_to_oversample]}, {subclass}")

                X_oversampled, y_oversampled, _ = self._one_class_oversample(X, y, instance_types, algorithm,
                                                                             all_classes,
                                                                             class_number_to_oversample, subclass,
                                                                             majority_class_name, random_state)

                scores = np.append(scores, run_cross_val(get_classifier(classifier_name, random_state), X_oversampled, y_oversampled))

            self.logger.debug(f"Scores: {scores}")

            highest_scores = np.argwhere(scores == np.amax(scores))
            if len(highest_scores) == 1:
                instance_types_to_oversample = highest_scores[0][0]
            elif 0 in highest_scores:
                instance_types_to_oversample = 0
            else:
                instance_types_to_oversample = highest_scores[np.random.randint(0, len(highest_scores))][0]

            instance_types_per_class.append(instance_types_to_oversample)

            X, y, oversampled = self._one_class_oversample(X, y, instance_types, algorithm, all_classes,
                                                           class_number_to_oversample,
                                                           all_subclasses[instance_types_to_oversample],
                                                           majority_class_name, random_state)
            oversampled_classes.append(oversampled)

            self.logger.debug(f"Instance types to oversample: {instance_types_to_oversample}")
        self.logger.debug(f"Final class count: {get_class_count(y)}")
        self.logger.debug(f"Oversampled classes: {oversampled_classes}")

        return X, y, [all_subclasses[type_number] for type_number in instance_types_per_class], oversampled_classes
