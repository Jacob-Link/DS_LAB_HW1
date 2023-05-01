# transform features as you wish

import pandas as pd


def std_mean_transform(patient_dict):
    """
    :param patient_dict: {"label": 0/1, "df": DataFrame}
    :return: list built from 3 parts, 1) col mean list, 2) col std list, 3) label
    """
    # for each col calc std and mean, use both metrics to define column
    # TODO: can choose to exclude constant columns like age and gender, for them use only the column value
    mean_array = list(patient_dict["df"].mean(axis=0))
    std_array = list(patient_dict["df"].std(axis=0))
    label = [patient_dict["label"]]
    if label == 1:
        print("ONE!!!")
    row = mean_array + std_array + label
    return row

