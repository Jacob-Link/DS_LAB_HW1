# transform features as you wish
import numpy as np
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
    row = mean_array + std_array + label
    return row


def impute_mean(df, df_name):
    # data is in nd array, cant use col function
    col_mean = np.nanmean(df, axis=0)
    inds = np.where(np.isnan(df))
    df[inds] = np.take(col_mean, inds[1])
    print(f">>> imputed mean for nan values in {df_name}")
    return df
