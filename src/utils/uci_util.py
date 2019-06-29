import sys
import urllib
from configparser import ConfigParser

import pandas as pd

from utils import logging_util

class UciConnector:

    logger = logging_util.get_new_logger(__name__)

    def __init__(self):
        pass

    def get_dataset(self, dataset_name: str):

        dataset_names = ConfigParser()
        dataset_names.read('../config/datasets.ini')

        try:
            with open(f"../data/datasets/{dataset_name}.csv") as arff_file:
                dataset = pd.read_csv(arff_file, header=None)
            return dataset, len(dataset.columns) - 1

        except urllib.error.HTTPError as e:
            self.logger.critical(e)
            self.logger.critical(f"Unable to find the dataset {dataset_name}")
            sys.exit(1)
