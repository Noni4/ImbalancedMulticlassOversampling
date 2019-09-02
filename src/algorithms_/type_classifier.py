import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from instance_types import Types
from math import log, floor

import numpy as np

class StaticKNNTypeClassifier:

    def __init__(self):
        pass

    def get_types(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        types = []
        neibors = knn.kneighbors(X, n_neighbors=6, return_distance=False) #TODO correct spelling mistake
        actual_neibors = neibors[:,1:] # counts it self its own neighbor
        for i in range(len(actual_neibors)):
            instance_neighbors_of_the_same_class = 0
            for neibor in actual_neibors[i]:
                if y[i] == y[neibor]:
                    instance_neighbors_of_the_same_class += 1
            if instance_neighbors_of_the_same_class >= 4:
                types.append(Types.SAFE)
                continue
            elif instance_neighbors_of_the_same_class >= 2:
                types.append(Types.BORDERLINE)
                continue
            elif instance_neighbors_of_the_same_class >= 1:
                types.append(Types.RARE)
                continue
            else:
                types.append(Types.OUTLIER)
        return np.array(types)
