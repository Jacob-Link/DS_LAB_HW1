# import relevant libs
# arg - path to test data
# clean data - for the ml model (drop all redundent ones
# transform the data
# load pickle from env
# predict on the test path data

import sys
import pandas as pd
import numpy as np
import os


def get_path():
    # test_path = sys.argv[1]
    test_path = r"C:\Users\Jacob Link\Desktop\Data_Science_Engineer\Year_3_Part_2\Lab in data science\HW\HW1\DS_LAB_HW1\data/test/"
    return test_path


# ---------------------------- SELECTION FUNCTION ---------------------------
def keep_all(patient_dict):
    return patient_dict["df"]


# ------------------------- TRANSFORMATION FUNCTION --------------------------
def std_mean_transform(patient_dict):
    """
    :param patient_dict: {"label": 0/1, "df": DataFrame}
    :return: list built from 3 parts, 1) col mean list, 2) col std list, 3) label
    """
    # for each col calc std and mean, use both metrics to define column
    mean_array = list(patient_dict["df"].mean(axis=0))
    std_array = list(patient_dict["df"].std(axis=0))
    label = [patient_dict["label"]]
    row = mean_array + std_array + label
    return row


def load_all_patients_for_ml_model(path):
    all_files = os.listdir(path)
    df_dict = dict()
    for i, f in enumerate(all_files):
        if (i + 1) % 500 == 0:
            print(f">>> loaded [{i + 1:,}/{len(all_files):,}] patients data...")
        df = pd.read_csv(path + f"/{f}", sep="|")
        id = f[:-4]
        df_dict[id] = df

    print(f">>> Total of {len(df_dict):,} patients files loaded successfully")
    return df_dict


def get_relevant_rows(group):
    # for each group will find the first row with label eq to 1 and return all rows including the first appearance of 1
    if group["SepsisLabel"].sum():
        group = group.reset_index(drop=True)
        return group.iloc[:group.loc[group["SepsisLabel"] == 1].index[0] + 1, :]
    else:
        return group


def modify_dfs(dfs_dict):
    # modification: remove label column, remove all rows eq 1 except 1 for those labeled 1.
    # return dict of keys: patient id,values: dict of label and df ready to input to ml model
    for patient, df in dfs_dict.items():
        df = get_relevant_rows(df)
        dfs_dict[patient] = {"label": sum(df["SepsisLabel"]), "df": df.drop(columns=["SepsisLabel"])}
    return dfs_dict


def load_test_data(test_path):
    dict_dfs = load_all_patients_for_ml_model(test_path)
    test_dict_dfs = modify_dfs(dict_dfs)
    return test_dict_dfs


def commit_selection(train_dict, selection_func):
    for patient, patient_dict in train_dict.items():
        train_dict[patient]["df"] = selection_func(patient_dict)
    return train_dict


def commit_row_transformation(train_dict, transformation_func):
    """
    :param train_dict: patient_id: {label: 0/1, df: DataFrame}
    :param transformation_func: function imported from the transformation py file
    :return: rows of all the transformed data if the transformation is an aggregation of df to row for each patient
    """
    data_rows = []
    for patient_dict in train_dict.values():
        data_rows.append(transformation_func(patient_dict))
    return data_rows


def split_matrix(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def commit_imputation(X_test):
    X_test = np.nan_to_num(X_test)
    return X_test


def x_y_test(path):
    # Feature selection chosen: "keep all"
    # Feature transformation chosen: "std mean"
    # Imputation: "Zero-imputation"

    data_dict = load_test_data(path)
    data_dict_after_selection = commit_selection(data_dict, keep_all)
    data_after_transformation = commit_row_transformation(data_dict_after_selection, std_mean_transform)
    test_matrix = np.array(data_after_transformation)

    X_test, y_test = split_matrix(test_matrix)
    X_test = commit_imputation(X_test)

    return X_test, y_test


if __name__ == '__main__':
    test_path = get_path()
    model_name = "xgboost_0714.pkl"

    X_test, y_test = x_y_test(test_path)
    print("Loaded everything successfully")
    input()

    # model = load_model(model_name)
    # # validation score
    # predictions = model.predict(X_test)
    # print("Validation score:")
    # f1 = calc_f1(predictions, y_test)
    # # export_prediction_csv()
    #
    # input()
