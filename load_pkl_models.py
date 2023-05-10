from model_evaluation import x_y_test, x_y_train, calc_f1, calc_precision, calc_recall
import pickle

from feature_selection import top_ten_non_missing, keep_all
from feature_transformation import std_mean_transform, impute_mean


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
        print(f'>>> successfully loaded {file_name}')
    return model


if __name__ == '__main__':
    # model_name = "gradient_boosting_071.pkl"
    # model_name = "random_forest_0676.pkl"
    # model_name = "logistic_regression_0403.pkl"
    # model_name = "adaboost_0694.pkl"
    # model_name = "lightGBM_0705.pkl"
    # model_name = "xgboost_0714.pkl"

    selection = keep_all
    transformation = std_mean_transform

    X_train, y_train = x_y_train(selection, transformation)
    X_test, y_test = x_y_test(selection, transformation)

    for model_name in ["xgboost_0714.pkl", "lightGBM_0705.pkl", "adaboost_0694.pkl", "random_forest_0676.pkl",
                       "gradient_boosting_071.pkl"]:
        model = load_model(model_name)
        # train score
        predictions = model.predict(X_train)
        print(f'\n{model_name}: ')
        print("Training score:")
        calc_f1(predictions, y_train)
        calc_precision(predictions, y_train)
        calc_recall(predictions, y_train)

        # validation score
        predictions = model.predict(X_test)
        print("Validation score:")
        calc_f1(predictions, y_test)
        calc_precision(predictions, y_test)
        calc_recall(predictions, y_test)
