from filelock import FileLock
from configparser import ConfigParser

from utils.types_util import types_to_string_list

config = ConfigParser()
config.read('../config/config.ini')

def write_result(dataset_name, type_algorithm, class_number, type_subset, seed, result, oversampled_classes, classifier_name, oversampling_algorithm):
    file_path = f"../data/results/{config['Files']['results']}{classifier_name}{oversampling_algorithm}"

    # with FileLock(file_path):
    with open(file_path, 'a') as result_file:
        result_file.write(f"{dataset_name};{type_algorithm};{class_number};{types_to_string_list(type_subset)};{seed};"
                          f"{result};{oversampled_classes}\n")