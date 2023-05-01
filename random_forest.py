import numpy as np

from cleaning_data import load_train_data_for_ml_model
from feature_selection import top_ten_non_missing
from feature_transformation import std_mean_transform

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


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


def create_rf_model(X, y):
    clf = RandomForestClassifier(max_depth=None, random_state=0)
    clf.fit(X, y)
    print(">>> successfully created random forest classifier")
    return clf


if __name__ == '__main__':
    data_dict = load_train_data_for_ml_model(load_pickle=True)
    data_dict_after_selection = commit_selection(data_dict, top_ten_non_missing)
    data_after_transformation = commit_row_transformation(data_dict_after_selection, std_mean_transform)
    data_matrix = np.array(data_after_transformation)

    # TODO: the nans are from the transformation, when theres only 0 or 1 values in column transformed
    data_matrix = np.nan_to_num(data_matrix)

    X, y = split_matrix(data_matrix)
    random_forest_model = create_rf_model(X, y)
