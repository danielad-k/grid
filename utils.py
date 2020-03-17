import itertools

import pandas as pd


def param_dict_convert(param_dictionary):
    """
    Converts dictionary of lists into lists of dictionary with single key,value pair

    :return: returns dictionary with key, value
    :param: param_dictionary: original dictionary with lists
    """
    return [dict(zip(param_dictionary, i)) for i in (itertools.product(*param_dictionary.values()))]


def convert_coherence_df(dictionary):
    return pd.DataFrame.from_dict(dictionary).sort_values(by='coherence', ascending=False)
