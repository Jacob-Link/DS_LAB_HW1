import pandas as pd
import numpy as np

from cleaning_data import load_train_data_for_ml_model


def missing_all_values(patient_dict):
    # if at least 1 value in column is not nan - output for column will be True
    row = list(~patient_dict["df"].isna().all(axis=0))
    return row


def boolean_df_is_all_missing(data_dict):
    # creates df of row for each patient, True if theres at least 1 value to aggregate in feature transformation
    rows = []
    for patient in data_dict:
        row = missing_all_values(data_dict[patient])
        rows.append(row)
    df = pd.DataFrame(data=rows, columns=data_dict[patient]["df"].columns)
    return df


def export(df):
    df.to_csv("at_least_one_val_per_column_per_patient.tsv", sep="\t")
    print(">>> successfully exported df")


if __name__ == '__main__':
    data_dict = load_train_data_for_ml_model(load_pickle=True)
    df = boolean_df_is_all_missing(data_dict)
    export(df)
    # rows = [[True, False, True], [False, False, True]]
    # rows = [[np.nan, 10, np.nan], [np.nan, 1, np.nan]]
    # df = pd.DataFrame(data=rows, columns=["A", "B", "C"])
    # print(df)
    # for
    # missing_all_values({"df": df, "label": 1})
    # print(df)
