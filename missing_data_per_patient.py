import pandas as pd
import numpy as np

from cleaning_data import load_train_data_for_ml_model


def missing_all_values_check(patient_dict):
    # if at least 1 value in column is not nan - output for column will be True
    row = list(~patient_dict["df"].isna().all(axis=0))
    return row


def boolean_df_is_all_missing(data_dict):
    # creates df of row for each patient, True if theres at least 1 value to aggregate in feature transformation
    rows = []
    for patient in data_dict:
        row = missing_all_values_check(data_dict[patient])
        rows.append(row)
    df = pd.DataFrame(data=rows, columns=data_dict[patient]["df"].columns)
    return df


def get_age_gender(patient_dict):
    patient_df = patient_dict["df"]
    age = list(patient_df["Age"].unique())
    if len(age) != 1:
        raise print("More then 1 value for Age column")
    gender = list(patient_df["Gender"].unique())
    if len(gender) != 1:
        raise print("More then 1 value for Gender column")

    return [age[0], gender[0]]


def get_age_gender_per_patient_df(data_dict):
    rows = []
    for patient in data_dict:
        row = get_age_gender(data_dict[patient])
        rows.append(row)
    df = pd.DataFrame(data=rows, columns=["Age", "Gender"])
    return df


def export(df):
    df.to_csv("at_least_one_val_per_column_per_patient.tsv", sep="\t")
    print(">>> successfully exported df")


if __name__ == '__main__':
    check_missing_columns = False
    age_gender_per_patient = True

    if check_missing_columns:
        data_dict = load_train_data_for_ml_model(load_pickle=True)
        df = boolean_df_is_all_missing(data_dict)
        export(df)

    if age_gender_per_patient:
        data_dict = load_train_data_for_ml_model(load_pickle=True)
        df = get_age_gender_per_patient_df(data_dict)
        print(df["Age"].describe())
        print()
        print(df["Gender"].describe())
