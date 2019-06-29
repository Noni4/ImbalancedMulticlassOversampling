from collections import Counter

import numpy as np
from collections import OrderedDict
from scipy.stats import ttest_ind, ttest_ind_from_stats


def to_percent(value):
    return value * 100

classifier = 'KNN'
algorithm_name = 'Dependent biggest to smallest'
first = np.loadtxt(f'../data/results/dependedlocalOptimized{classifier}', delimiter=';', dtype=object)
second = np.loadtxt(f'../data/results/dependedlocalOptimizedReversed{classifier}', delimiter=';', dtype=object)

vs_all = False

if vs_all:

    datasets = first[:,0]
    algorithms = ['independent locally optimized', 'dependent locally optimized (biggest first)']
    subsets = second[:,3]

    datasets = list(np.unique(datasets))
    algorithms = list(np.unique(algorithms))
    subsets = list(np.unique(subsets))

    print(f"Subsets: {subsets}")

    matrix = np.array([[None] * (len(subsets) + 2)] * (len(datasets) + 1), dtype=object)
    print(matrix.shape)

    matrix[0][0] = 'Datasets'
    for i in range(len(datasets)):
        matrix[i+1][0] = datasets[i]
        if i == 0:
            matrix[i][1] = 'locally optimized'
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
            if i == 0 and j < 10 or i > 1 and j == 0:
                result_string += f"{matrix[i][j]} & "
            elif j == 1 and i > 1:
                result_string += f"${round(to_percent(matrix[i][j][0]), 2)}$\\pm${round(to_percent(matrix[i][j][1]), 2)}$ & "
            elif j < 10 and j > 1 and i > 1:
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
    result_string += ' , Static Types, '
    result_string += algorithm_name
    result_string += '} \n \\label{tab:my_label} \n \\end{table} \n'
    print(result_string)

    result_string = ''
    result_string = '\\begin{table}[t] \n \\centering \n \\resizebox{\\textwidth}{!}{ \n \\begin{tabular}{c|c|c|c|c|c|c|c|c|c}'
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == 0 and (j >= 10 or j == 0 or j == 1) or i > 1 and j == 0:
                result_string += f"{matrix[i][j]} & "
            elif j == 1 and i > 1:
                result_string += f"${round(to_percent(matrix[i][j][0]), 2)}$\\pm${round(to_percent(matrix[i][j][1]), 2)}$ & "
            elif j >= 10 and j > 1 and i > 1:
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
    result_string += ' , Static Types, '
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

    result_string = '\\begin{table}[t] \n \\centering \n \\begin{tabular}{c|c} \n Subset & Occurrences \\\\ \n'
    subset_count = OrderedDict(sorted(subset_count.items(), key=lambda kv: kv[1], reverse=True))

    for x, y in subset_count.items():
        result_string += f"{x} & ${y}$ \\\\ \n".replace('[', '').replace(']', '').replace("'", '')
    result_string += '\\end{tabular} \n \\caption{'
    result_string += classifier
    result_string += ' , Static Types, '
    result_string += algorithm_name
    result_string += '} \n \\label{tab:my_label} \n \\end{table} \n'

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
            matrix[i][1] = 'dependent locally optimized (biggest first)'
            matrix[i][2] = 'dependent locally optimized (smallest first)'
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
            if i == 0 and j < 10 or i > 1 and j == 0:
                result_string += f"{matrix[i][j]} & "
            elif j == 1 and i > 1:
                result_string += f"${round(to_percent(matrix[i][j][0]), 2)}$\\pm${round(to_percent(matrix[i][j][1]),2)}$ & "
            elif j < 10 and j > 1 and i > 1:
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
    result_string += ', Static Types} \n \\label{tab:my_label} \n \\end{table}'

print(result_string)