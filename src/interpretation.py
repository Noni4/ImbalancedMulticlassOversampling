import csv

import numpy as np

from tabulate import tabulate

from scipy.stats import ttest_ind

with open('../data/firstrun', 'r') as csv_file:
    data = list(csv.reader(csv_file, delimiter=';'))

subsets = set()
datasets = set()
number_of_classes = set()

for row in data:
    subsets.add(row[3])
    datasets.add(row[0])
    number_of_classes.add(row[2])

one_class_oversampling = 0
static_types = 0

for subset in subsets:
    static_local = 0
    results = []
    for dataset in datasets:
        if True:
            static = []
            dynamic = []

            for row in data:
                if row[1] == 'static' and row[3] == subset and dataset == row[0] and 'one' == row[2]:
                    static.append(float(row[5]))
                if row[1] == 'dynamic' and row[3] == subset and dataset == row[0] and 'one' == row[2]:
                    dynamic.append(float(row[5]))

            ps = ''
            if ttest_ind(static, dynamic)[1] < 0.1:
                if ttest_ind(static, dynamic)[0] > 0:
                    ps = 'static'
                else:
                    ps = 'dyn'
            else:
                ps = 'None'

            sta = np.mean(np.array(static))
            dyn = np.mean(np.array(dynamic))

            if sta > dyn:
                static_types += 1
                static_local += 1
            if dyn > sta:
                static_types -= 1
                static_local -=1

            result = []
            result.append(subset)
            result.append(dataset)
            result.append(sta)
            result.append(dyn)
            result.append(ps)
            results.append(result)

    print(tabulate(results))
    print(static_local)

print(static_types)