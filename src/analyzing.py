from typing import Dict

import numpy as np
import pandas as pd


class Analyzer:

    def __init__(self):
        pass

    def get_class_occurences(self, dataset: pd.DataFrame) -> pd.Series:
        return dataset.iloc[:, 0].value_counts()

    def get_number_of_instances_to_oversample(self, dataset: pd.DataFrame) -> Dict:
        class_occurences = self.get_class_occurences(dataset)
        number_of_instances_to_oversample = {}
        number_of_instances_to_oversample[str(class_occurences.iloc[[0]].index[0])] = 0
        for i in range(1, len(class_occurences)): # [0] is always the majority class
            number_of_instances_to_oversample[str(class_occurences.iloc[[i]].index[0])] = class_occurences.iloc[0] - class_occurences.iloc[i]
        return number_of_instances_to_oversample