import numpy as np
import pickle

from cleaning_data import load_train_data_for_ml_model, load_test_data
from feature_selection import top_ten_non_missing, keep_all, forward_sfs, chat_GPT_selection, drop_high_correlated
from feature_transformation import std_mean_transform, impute_mean

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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


def create_gradient_boosting_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    clf = GradientBoostingClassifier(n_estimators=1000)
    clf.fit(X, y)
    print(">>> successfully created random forest classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def create_logistic_regression_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    clf = LogisticRegression(max_iter=1000, solver="newton-cg")
    clf.fit(X, y)
    print(">>> successfully created Logistic Regression classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def x_y_train(selection, transformation):
    # load train data
    data_dict = load_train_data_for_ml_model(load_pickle=True)
    data_dict_after_selection = commit_selection(data_dict, selection)
    data_after_transformation = commit_row_transformation(data_dict_after_selection, transformation)
    train_matrix = np.array(data_after_transformation)

    X_train, y_train = split_matrix(train_matrix)

    # TODO: the nans are from the transformation, when theres only 0 or 1 values in column transformed
    # X_train = impute_mean(X_train, "X_train df")
    X_train = np.nan_to_num(X_train)

    return X_train, y_train


def x_y_test(selection, transformation):
    # load test data
    data_dict = load_test_data(load_pickle=True)
    data_dict_after_selection = commit_selection(data_dict, selection)
    data_after_transformation = commit_row_transformation(data_dict_after_selection, transformation)
    test_matrix = np.array(data_after_transformation)

    X_test, y_test = split_matrix(test_matrix)

    # TODO: the nans are from the transformation, when theres only 0 or 1 values in column transformed
    # X_test = impute_mean(X_test, "X_train df")
    X_test = np.nan_to_num(X_test)

    return X_test, y_test


def create_rf_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    clf = RandomForestClassifier(n_estimators=1000, criterion="gini")
    clf.fit(X, y)
    print(">>> successfully created random forest classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def create_adaboost_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    clf = AdaBoostClassifier(n_estimators=1000)
    clf.fit(X, y)
    print(">>> successfully created AdaBoost classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def create_xgboost_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    # clf = XGBClassifier(n_estimators=2000, tree_method='hist')
    clf = XGBClassifier(n_estimators=2000, tree_method='hist', max_depth=5)
    clf.fit(X, y)
    print(">>> successfully created XGBoost classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def create_lgbm_model(X, y, export_pkl=False, model_file_name=None):
    # return a fitted model
    clf = LGBMClassifier(n_estimators=2000)
    clf.fit(X, y)
    print(">>> successfully created LightGBM classifier")
    if export_pkl:
        with open(model_file_name, 'wb') as f:
            pickle.dump(clf, f)
            print(f'>>> successfully exported {model_file_name}')
    return clf


def calc_f1(predictions, y_test):
    f1 = f1_score(y_test, predictions)
    print(f"F1 score: {round(f1, 3)}")
    return f1


def get_model(model_name, X_train, y_train, export=False):
    if model_name == "rf":
        random_forest_model = create_rf_model(X_train, y_train, export_pkl=export,
                                              model_file_name="random_forest_.pkl")
        return random_forest_model

    if model_name == "gb":
        gradient_boosting_model = create_gradient_boosting_model(X_train, y_train, export_pkl=export,
                                                                 model_file_name="gradient_boosting_.pkl")
        return gradient_boosting_model

    if model_name == "lr":
        logistic_regression_model = create_logistic_regression_model(X_train, y_train, export_pkl=export,
                                                                     model_file_name="logistic_regression_.pkl")
        return logistic_regression_model

    if model_name == "ada":
        adaboost_model = create_adaboost_model(X_train, y_train, export_pkl=export,
                                               model_file_name="adaboost_.pkl")
        return adaboost_model

    if model_name == "xgb":
        xgboost_model = create_xgboost_model(X_train, y_train, export_pkl=export,
                                             model_file_name="xgboost_.pkl")
        return xgboost_model

    if model_name == "lgbm":
        lgbm_model = create_lgbm_model(X_train, y_train, export_pkl=export,
                                       model_file_name="lightGBM_.pkl")
        return lgbm_model


def check_forward_stepwise(X_train, y_train):
    model = RandomForestClassifier(n_estimators=1000)
    bool_mask, selected_new_x = forward_sfs(model, X_train, y_train)
    return bool_mask, selected_new_x


def step_wise_forward(X_train, y_train, X_test, y_test):
    bool_mask, new_X = check_forward_stepwise(X_train, y_train)
    print("*****************************")
    print(bool_mask)
    print("*****************************")
    model = get_model("xgb", new_X, y_train, export=False)
    predictions = model.predict(X_test[:, bool_mask])
    f1 = calc_f1(predictions, y_test)


if __name__ == '__main__':
    # define selection and transformation at beginning
    selection = keep_all
    transformation = std_mean_transform

    X_train, y_train = x_y_train(selection, transformation)
    X_test, y_test = x_y_test(selection, transformation)

    # options: "rf", "gb", "lr", "ada", "xgb", "lgbm"
    model = get_model("xgb", X_train, y_train, export=False)

    # train score
    predictions = model.predict(X_train)
    print("Training score:")
    f1 = calc_f1(predictions, y_train)

    # validation score
    predictions = model.predict(X_test)
    print("Validation score:")
    f1 = calc_f1(predictions, y_test)
