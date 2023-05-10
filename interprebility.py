from cleaning_data import load_test_data
from load_pkl_models import load_model
import pandas as pd


def get_feature_names():
    data_dict = load_test_data(load_pickle=True)
    for patient in data_dict:
        raw_ft = list(data_dict[patient]["df"].columns)
        break
    mean_ft = [x + "_mean" for x in raw_ft]
    std_ft = [x + "_std" for x in raw_ft]
    return mean_ft + std_ft


def print_importance(model, model_name):
    feature_importance = model.feature_importances_
    feature_names = get_feature_names()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df.sort_values("Importance", inplace=True, ascending=False)
    importance_df["Importance"] = importance_df["Importance"].round(5)
    print(f"{model_name}")
    print(importance_df)
    importance_df.to_csv(f"{model_name[:-9]}_feature_importance.csv", index=False)


if __name__ == '__main__':
    for model_name in ["xgboost_0714.pkl", "adaboost_0694.pkl", "random_forest_0676.pkl"]:
        model = load_model(model_name)
        print_importance(model, model_name)
