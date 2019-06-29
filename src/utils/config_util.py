from configparser import ConfigParser

from itertools import chain, combinations

config = ConfigParser()
config.read('../config/config.ini')

def get_experiment_configuration(key):
    return config['Experiment'][key].split(',')

def powerset(set_: iter):
    if 'None' in set_:
        return set('0')
    return chain.from_iterable(combinations(set_, subset) for subset in range(len(set_) + 1))