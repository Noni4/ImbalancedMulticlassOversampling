from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pandas as pd
import numpy as np

from utils import logging_util

class Preprocessor:

    logger = logging_util.get_new_logger(__name__)

    def __int__(self):
        pass

    def encode(self, X: np.ndarray):
        row_numbers_to_encode = set()
        for i in range(len(X)):
            for j in range(len(X[i])):
                try:
                    float(X[i][j])
                except ValueError:
                    row_numbers_to_encode.add(j)

        row_numbers_to_encode_list = list(row_numbers_to_encode)

        X = pd.DataFrame(X)

        self.logger.debug(f"Unencoded dataset: {X}")

        if len(row_numbers_to_encode_list) > 0:
            rows_to_encode = X.iloc[:, row_numbers_to_encode_list]
            label_encoder = LabelEncoder()
            labeled_rows = rows_to_encode.apply(label_encoder.fit_transform)

            one_hot_encoder = OneHotEncoder(categories='auto')
            encoded_rows = pd.DataFrame(one_hot_encoder.fit_transform(labeled_rows).toarray())
            X = X.drop(X.iloc[:, row_numbers_to_encode_list], axis=1)
            X: pd.DataFrame = pd.concat([X, encoded_rows], axis=1, sort=False)

        self.logger.debug(f"Encoded datasset: {X}")

        return np.asarray(X.values, dtype=np.float64)