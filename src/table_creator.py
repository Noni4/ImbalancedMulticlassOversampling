"""This class was rewritten multuple times to display different evaluations

"""

from collections import Counter

import numpy as np
import re
from collections import OrderedDict
from scipy.stats import ttest_ind, ttest_ind_from_stats

from algorithms_.preprocessing import Preprocessor
from algorithms_.type_classifier import StaticKNNTypeClassifier
from utils.config_util import get_experiment_configuration
from utils.ml_util import dataframe_to_arrays
from utils.uci_util import UciConnector
from utils.types_util import Types


def to_percent(value):
    return value * 100

def result_table():
    classifier = 'SVM'
    algorithm_name = 'Dependent Locally Optimized (Smallest first)'
    first = np.loadtxt(f'../data/results/newBiggestFirstDependentLocallyOptimized{classifier}SMOTE', delimiter=';',
                       dtype=object)
    second = np.loadtxt(f'../data/results/newSmallestFirstDependentLocallyOptimized{classifier}SMOTE', delimiter=';',
                        dtype=object)
    vs_all = False

    if vs_all:

        datasets = first[:,0]
        algorithms = ['independent locally optimized', 'dependent locally optimized (biggest first)']
        subsets = second[:,3]

        datasets = list(np.unique(datasets))
        # algorithms = list(np.unique(algorithms))
        subsets = list(np.unique(subsets))

        print(f"Subsets: {subsets}")

        matrix = np.array([[None] * (len(subsets) + 2)] * (len(datasets) + 1), dtype=object)
        print(matrix.shape)

        matrix[0][0] = 'Datasets'
        for i in range(len(datasets)):
            matrix[i+1][0] = datasets[i]
            if i == 0:
                matrix[i][1] = 'smallest first dependent locally optimized'
                for j in range(len(subsets)):
                    matrix[i][j+2] = 'all('
                    if 'safe' in subsets[j]:
                        matrix[i][j+2] += 'S'
                    if 'borderline' in subsets[j]:
                        matrix[i][j+2] += 'B'
                    if 'rare' in subsets[j]:
                        matrix[i][j+2] += 'R'
                    if 'outlier' in subsets[j]:
                        matrix[i][j+2] += 'O'
                    matrix[i][j+2] += ')'
            for algorithm in algorithms:
                if 'optimized' in algorithm.lower():
                    local_results = []
                    for row in first:
                        if row[0] == datasets[i]:
                            local_results.append(float(row[5]))
                    matrix[i+1][1] = (np.mean(local_results), np.std(local_results))
                    # print(f"Adding: {matrix[i][0]} for {algorithm}")
                else:
                    for j in range(len(subsets)):
                        local_results = []
                        for row in second:
                            if row[0] == datasets[i] and row[3] == subsets[j]:
                                local_results.append(float(row[5]))
                        if len(local_results) == 0:
                            raise ZeroDivisionError
                        matrix[i+1][j+2] = (np.mean(local_results), np.std(local_results))
                        # print(f"Adding: {matrix[i][j+1]} for {subsets}, {algorithm}")

        print(f"Matrix shape: {matrix.shape}")
        print(matrix)

        result_string = ''
        result_string = '\\begin{table}[t] \n \\centering \n \\resizebox{\\textwidth}{!}{ \n \\begin{tabular}{c|c|c|c|c|c|c|c|c|c} \n'
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == 0 and j < 10 or i >= 1 and j == 0:
                    result_string += f"{matrix[i][j]} & "
                elif j == 1 and i >= 1:
                    result_string += f"${round(to_percent(matrix[i][j][0]), 2)}$\\pm${round(to_percent(matrix[i][j][1]), 2)}$ & "
                elif j < 10 and j > 1 and i >= 1:
                    t, p = ttest_ind_from_stats(matrix[i][j][0], matrix[i][j][1], 20, matrix[i][1][0], matrix[i][1][1], 20)
                    if p < 0.05 and t < 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]), 2)}\\circ$ & "
                    elif p < 0.05 and t > 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]), 2)}\\bullet$ & "
                    else:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]), 2)}$ & "
            result_string = result_string[:-2]
            result_string += '\\\\'
            result_string += '\n'
        result_string += '\\end{tabular}} \n \\caption{'
        result_string += classifier
        result_string += ', '
        result_string += algorithm_name
        result_string += '} \n \\label{tab:my_label} \n \\end{table} \n'
        print(result_string)

        result_string = ''
        result_string = '\\begin{table}[t] \n \\centering \n \\resizebox{\\textwidth}{!}{ \n \\begin{tabular}{c|c|c|c|c|c|c|c|c|c} \n'
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == 0 and (j >= 10 or j == 0 or j == 1) or i >= 1 and j == 0:
                    result_string += f"{matrix[i][j]} & "
                elif j == 1 and i >= 1:
                    result_string += f"${round(to_percent(matrix[i][j][0]), 2)}$\\pm${round(to_percent(matrix[i][j][1]), 2)}$ & "
                elif j >= 10 and j > 1 and i >= 1:
                    t, p = ttest_ind_from_stats(matrix[i][j][0], matrix[i][j][1], 20, matrix[i][1][0], matrix[i][1][1], 20)
                    if p < 0.05 and t < 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]), 2)}\\circ$ & "
                    elif p < 0.05 and t > 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]), 2)}\\bullet$ & "
                    else:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]),2)}$ & "
            result_string = result_string[:-2]
            result_string += '\\\\'
            result_string += '\n'
        result_string += '\\end{tabular}} \n \\caption{'
        result_string += classifier
        result_string += ', '
        result_string += algorithm_name
        result_string += '} \n \\label{tab:my_label} \n \\end{table} \n'
        print(result_string)

        subsets_rows = first[:,3]
        subsets = set(subsets)

        subset_count = {}
        for subset in subsets:
            subset_count[subset] = 0
            subset_list = subset.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').split(',')
            for row in subsets_rows:
                row = row.split('},')
                for elem in row:
                    elem = elem.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(' ', '').replace("'", '').split(',')
                    if set(elem) == set(subset_list):
                        subset_count[subset] = subset_count[subset] + 1

        result_string = '\\begin{table}[t] \n \\centering \n \\begin{tabular}{c|c} \n Subset & Occurrences \\\\ \\hline \n'
        subset_count = OrderedDict(sorted(subset_count.items(), key=lambda kv: kv[1], reverse=True))

        subset_sum = 0
        for x, y in subset_count.items():
            subset_sum += y

        for x, y in subset_count.items():
            result_string += f"{x} & ${round(y/subset_sum * 100, 3)}\\%$ \\\\ \\ \n".replace("'", '')
        result_string += '\\end{tabular} \n \\caption{'
        result_string += classifier
        result_string += ' , Static Types, '
        result_string += algorithm_name
        result_string += '} \n \\label{tab:my_label} \n \\end{table} \n'

        print(subset_sum)

    else:
        datasets = first[:, 0]
        algorithms = ['localOptimized', 'independedlocallyOptimized']

        datasets = list(np.unique(datasets))
        algorithms = list(np.unique(algorithms))

        matrix = np.array([[None] * 3] * (len(datasets) + 1), dtype=object)

        matrix[0][0] = 'Datasets'
        for i in range(len(datasets)):
            matrix[i + 1][0] = datasets[i]
            if i == 0:
                matrix[i][1] = 'DLO (biggest first)'
                matrix[i][2] = 'DLO (smallest first)'
            for algorithm in algorithms:
                if algorithm == 'localOptimized':
                    local_results = []
                    for row in first:
                        if row[0] == datasets[i]:
                            local_results.append(float(row[5]))
                    matrix[i + 1][1] = (np.mean(local_results), np.std(local_results))
                else:
                    local_results = []
                    for row in second:
                        if row[0] == datasets[i]:
                            local_results.append(float(row[5]))
                    matrix[i + 1][2] = (np.mean(local_results), np.std(local_results))

        print(f"Matrix shape: {matrix.shape}")

        result_string = '\\begin{table}[t] \n \\centering \n \\begin{tabular}{c|c|c} \n'
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == 0 and j < 10 or i >= 1 and j == 0:
                    result_string += f"{matrix[i][j]} & "
                elif j == 1 and i >= 1:
                    result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]),2)}$ & "
                elif j < 10 and j > 1 and i >= 1:
                    t, p = ttest_ind_from_stats(matrix[i][j][0], matrix[i][j][1], 20, matrix[i][1][0], matrix[i][1][1], 20)
                    print(f"{matrix[i][0], matrix[i][j][0], matrix[i][j][1], 20, matrix[i][1][0], matrix[i][1][1], 20, t, p}")
                    if p < 0.05 and t < 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]),2)}\\circ$ & "
                    elif p < 0.05 and t > 0:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]),2)}\\bullet$ & "
                    else:
                        result_string += f"${round(to_percent(matrix[i][j][0]), 2)}\\pm{round(to_percent(matrix[i][j][1]),2)}$ & "
            result_string = result_string[:-2]
            result_string += '\\\\'
            result_string += '\n'
        result_string += '\\end{tabular} \n \\caption{'
        result_string += classifier
        result_string += '} \n \\label{tab:my_label} \n \\end{table}'

    print(result_string)

def types():
    type_algorithm = 'dynamic'

    # instances_types_per_class = []
    dataset_names = get_experiment_configuration('dataset')
    for dataset_name in dataset_names:

        uci_connector = UciConnector()
        preprocessor = Preprocessor()

        if type_algorithm == 'static':
            type_classifier = StaticKNNTypeClassifier()
        elif type_algorithm == 'dynamic':
            type_classifier = DynamicKNNTypeClassifier()

        dataset, class_index = uci_connector.get_dataset(dataset_name)

        X, y = dataframe_to_arrays(dataset, class_index)
        X = preprocessor.encode(X)

        instance_types = type_classifier.get_types(X, y)

        # print(dataset_name)
        # for class_ in OrderedDict(sorted(Counter(y).items(), key=lambda x: x[1], reverse=False)):
        #     print(class_)
        #     instances_types_per_class = []
        #     for i in range(len(y)):
        #         if class_ == y[i]:
        #             instances_types_per_class.append(instance_types[i])
            # counter_sum = Counter(instances_types_per_class).values()
            # print(f"{round(instances_types_per_class.count(Types.SAFE)/len(instances_types_per_class), 3)}/"
            #       f"{round(instances_types_per_class.count(Types.BORDERLINE)/len(instances_types_per_class), 3)}/"
            #       f"{round(instances_types_per_class.count(Types.RARE)/len(instances_types_per_class), 3)}/"
            #       f"{round(instances_types_per_class.count(Types.OUTLIER)/len(instances_types_per_class), 3)}")
            # print({key: value/counter_sum for key, value in sorted(Counter(instances_types_per_class).items(), key=lambda x: x[1], reverse=True)})

        print(dataset_name)
        instances_types_per_class = []
        for class_ in OrderedDict(sorted(Counter(y).items(), key=lambda x: x[1], reverse=False)):
            for i in range(len(y)):
                if class_ == y[i]:
                    instances_types_per_class.append(instance_types[i])
            # counter_sum = Counter(instances_types_per_class).values()
        print(f"{round(instances_types_per_class.count(Types.SAFE)/len(instances_types_per_class), 3)}/"
              f"{round(instances_types_per_class.count(Types.BORDERLINE)/len(instances_types_per_class), 3)}/"
              f"{round(instances_types_per_class.count(Types.RARE)/len(instances_types_per_class), 3)}/"
              f"{round(instances_types_per_class.count(Types.OUTLIER)/len(instances_types_per_class), 3)}")
            # print({key: value/counter_sum for key, value in sorted(Counter(instances_types_per_class).items(), key=lambda x: x[1], reverse=True)})



def barGraph():
    classifiers = ['DecisionTree', 'KNN', 'NaiveBayes', 'SVM']
    for classifier in classifiers:
        first = np.loadtxt(f'../data/results/newSmallestFirstDependentLocallyOptimized{classifier}SMOTE', delimiter=';',
                           dtype=object)
        second = np.loadtxt(f'../data/results/NotOptimized{classifier}SMOTE', delimiter=';',
                            dtype=object)

        vs_all = True

        all_first_results = dict()
        all_second_results = dict()

        datasets = first[:, 0]
        datasets = np.unique(datasets)

        for dataset in datasets:
            local_results = []
            for i in range(len(first)):
                if first[i, 0] == dataset:
                    local_results.append(float(first[i, 5]))
            all_first_results[dataset] = (np.mean(local_results), np.std(local_results))
        if vs_all:

            datasets = first[:, 0]
            subsets = second[:, 3]

            splitted_subsets = []
            for subset in subsets:
                for splitted_subset in subset.split('},'):
                    splitted_subset = splitted_subset.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '').replace("'", '').replace(' ', '')
                    if splitted_subset not in splitted_subsets:
                        splitted_subsets.append(splitted_subset)

            subsets = list(np.unique(subsets))

            for subset in subsets:
                subset = subset.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '').replace("'", '').replace(' ', '').split(',')
                second_results = dict()
                for dataset in datasets:
                    local_results = []
                    for i in range(len(second)):
                        if second[i, 0] == dataset:
                            data_subsets = second[i, 3].replace('[', '').replace(']', '').replace("'", '').replace(' ', '').split(',')
                            for subset_part in subset:
                                if subset_part in data_subsets:
                                    data_subsets.remove(subset_part)
                                else:
                                    break
                            else:
                                if len(data_subsets) == 0:
                                    local_results.append(float(second[i, 5]))
                    second_results[dataset] = (np.mean(local_results), np.std(local_results))
                all_second_results[str(subset)] = second_results

            final_results = dict()
            for subset in all_second_results:
                final_results[subset] = (0,0,0)
                for dataset in all_second_results[subset]:
                    t, p = ttest_ind_from_stats(all_first_results[dataset][0], all_first_results[dataset][1], 20,
                                                all_second_results[subset][dataset][0],
                                                all_second_results[subset][dataset][1], 20)
                    if p < 0.05 and t < 0:
                        final_results[subset] = (final_results[subset][0] + 1, final_results[subset][1], final_results[subset][2])
                        # print(f"{all_first_results[dataset][0]}/{all_first_results[dataset][1]} significantly worse than "
                        #       f"{all_second_results[subset][dataset][0]}/{all_second_results[subset][dataset][1]} at "
                        #       f"{subset}/{dataset}/{classifier}")
                    elif p < 0.05 and t > 0:
                        final_results[subset] = (final_results[subset][0], final_results[subset][1], final_results[subset][2] + 1)
                        # print(
                        #     f"{all_first_results[dataset][0]}/{all_first_results[dataset][1]} significantly better than "
                        #     f"{all_second_results[subset][dataset][0]}/{all_second_results[subset][dataset][1]} at "
                        #     f"{subset}/{dataset}/{classifier}")
                    else:
                        final_results[subset] = (final_results[subset][0], final_results[subset][1] + 1, final_results[subset][2])
                        # print(
                        #     f"{all_first_results[dataset][0]}/{all_first_results[dataset][1]} even to "
                        #     f"{all_second_results[subset][dataset][0]}/{all_second_results[subset][dataset][1]} at "
                        #     f"{subset}/{dataset}/{classifier}")
        else:
            pass

        all_subsets = ["['safe', 'borderline', 'rare', 'outlier']",
                       "['safe', 'borderline', 'rare']",
                       "['safe', 'borderline', 'outlier']",
                       "['safe', 'borderline']",
                       "['safe', 'rare', 'outlier']",
                       "['safe', 'rare']",
                       "['safe', 'outlier']",
                       "['safe']",
                       "['borderline', 'rare', 'outlier']",
                       "['borderline', 'rare']",
                       "['borderline', 'outlier']",
                       "['borderline']",
                       "['rare', 'outlier']",
                       "['rare']",
                       "['outlier']",
                       "['']"]

        synonyms = {"['safe', 'borderline', 'rare', 'outlier']": '[sbro]',
                       "['safe', 'borderline', 'rare']": '[sbr]',
                       "['safe', 'borderline', 'outlier']": '[sbo]',
                       "['safe', 'borderline']": '[sb]',
                       "['safe', 'rare', 'outlier']": '[sro]',
                       "['safe', 'rare']": '[sr]',
                       "['safe', 'outlier']": '[so]',
                       "['safe']": '[s]',
                       "['borderline', 'rare', 'outlier']": '[bro]',
                       "['borderline', 'rare']": '[br]',
                       "['borderline', 'outlier']": '[bo]',
                       "['borderline']": '[b]',
                       "['rare', 'outlier']": '[ro]',
                       "['rare']": '[r]',
                       "['outlier']": '[o]',
                       "['']": '[]'}
        print(classifier)
        for i in range(3):
            result_string = '\\addplot+[ybar] plot coordinates {'
            for subset in all_subsets:
                result_string += f"({synonyms[subset]}, {final_results[subset][i]}) "
            result_string = result_string[:-1] + '};'
            print(result_string)
        print('')

types()
