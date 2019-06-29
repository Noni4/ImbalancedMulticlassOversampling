from instance_types import Types


def types_to_string_list(types_list: list):
    string_list = []
    for types in types_list:
        if isinstance(types, tuple):
            string_set = set()
            if Types.SAFE in types:
                string_set.add('safe')
            if Types.BORDERLINE in types:
                string_set.add('borderline')
            if Types.RARE in types:
                string_set.add('rare')
            if Types.OUTLIER in types:
                string_set.add('outlier')
            string_list.append(string_set)
        else:
            if types == 'safe':
                string_list.append('safe')
            if types == 'borderline':
                string_list.append('borderline')
            if types == 'rare':
                string_list.append('rare')
            if types == 'outlier':
                string_list.append('outlier')
    return string_list

def string_list_to_types_list(string_list: list):
    types_list = []
    if 'safe' in string_list:
        types_list.append(Types.SAFE)
    if 'borderline' in string_list:
        types_list.append(Types.BORDERLINE)
    if 'rare' in string_list:
        types_list.append(Types.RARE)
    if 'outlier' in string_list:
        types_list.append(Types.OUTLIER)
    return types_list