from cleaning_data import load_test_data
import numpy as np
import pandas as pd
from feature_transformation import std_mean_transform
from model_evaluation import split_matrix
from load_pkl_models import load_model
from model_evaluation import calc_f1, calc_recall, calc_precision


def run_seperation_1(test_data_dict):
    age_bins = [0, 17, 29, 59, np.inf]
    age_labels = ['Under 18', '18-29', '30-59', '60+']

    # Create an empty dictionary to store the separated data
    separated_data = {age_label: {'male': [], 'female': []} for age_label in age_labels}

    for patient in test_data_dict:
        patient_df = test_data_dict[patient]['df']
        patient_age = patient_df.iloc[0]['Age']
        patient_gender = 'male' if patient_df.iloc[0]['Gender'] == 1 else 'female'

        # Find the age group the patient belongs to
        age_group = pd.cut([patient_age], bins=age_bins, labels=age_labels)[0]

        # Add the patient data to the appropriate age and gender group
        separated_data[age_group][patient_gender].append(test_data_dict[patient])
    datasets = {}
    for age_label in age_labels:
        datasets[age_label] = {}
        for gender in ['male', 'female']:
            datasets[age_label][gender] = separated_data[age_label][gender]

    # Print the number of samples in each group
    for age_label in age_labels:
        print(f"{age_label}:")
        for gender in ['male', 'female']:
            print(f"  {gender.capitalize()}: {len(separated_data[age_label][gender])} samples")

    return datasets


def run_seperation_2(test_data_dict):
    # Calculate the overall median values for Temp and MAP
    median_temp = pd.concat([test_data_dict[patient]['df']['Temp'].dropna() for patient in test_data_dict]).median()
    median_map = pd.concat([test_data_dict[patient]['df']['MAP'].dropna() for patient in test_data_dict]).median()

    # Create an empty dictionary to store the separated data
    separated_data = {'high_temp': {'high_map': [], 'low_map': []}, 'low_temp': {'high_map': [], 'low_map': []}}
    datasets = {'high_temp': {'high_map': [], 'low_map': []}, 'low_temp': {'high_map': [], 'low_map': []}}

    for patient in test_data_dict:
        patient_df = test_data_dict[patient]['df']
        patient_median_temp = patient_df['Temp'].dropna().median()
        patient_median_map = patient_df['MAP'].dropna().median()

        # Determine the patient's temperature group
        temp_group = 'high_temp' if patient_median_temp > median_temp else 'low_temp'

        # Determine the patient's MAP group
        map_group = 'high_map' if patient_median_map > median_map else 'low_map'

        # Add the patient ID to the appropriate group
        separated_data[temp_group][map_group].append(patient)
        datasets[temp_group][map_group].append(test_data_dict[patient])

    # Print the number of unique patient IDs in each group
    for temp_group in ['high_temp', 'low_temp']:
        print(f"{temp_group.capitalize()}:")
        for map_group in ['high_map', 'low_map']:
            print(f"  {map_group.capitalize()}: {len(set(separated_data[temp_group][map_group]))} patients")

    return datasets


def get_data_rows_after_transform(performance_test_data):
    data_rows = []
    for patient_dict in performance_test_data:
        data_rows.append(std_mean_transform(patient_dict))
    return data_rows


def run_age_gender():
    group = '30-59'
    test_data_dict = load_test_data(load_pickle=True)
    datasets = run_seperation_1(test_data_dict)

    performance_test_data = datasets[group]  # returns dict: key-male/female, value-list of patient dicts.

    data_dict = dict()
    for gender in performance_test_data:
        data_dict[gender] = dict()
        performance_test_data[gender] = get_data_rows_after_transform(performance_test_data[gender])
        train_matrix = np.array(performance_test_data[gender])
        data_dict[gender]["X_test"], data_dict[gender]["y_test"] = split_matrix(train_matrix)
        data_dict[gender]["X_test"] = np.nan_to_num(data_dict[gender]["X_test"])

    print(data_dict["male"]["X_test"])
    print(data_dict["male"]["y_test"])
    print()
    print(data_dict["female"]["X_test"])
    print(data_dict["female"]["y_test"])

    models = ["xgboost_0714.pkl", "adaboost_0694.pkl", "random_forest_0676.pkl"]
    for model_name in models:
        model = load_model(model_name)
        print(f"\n{model_name}")
        print("Male:")
        predictions = model.predict(data_dict["male"]["X_test"])
        f1 = calc_f1(predictions, data_dict["male"]["y_test"])
        precision = calc_precision(predictions, data_dict["male"]["y_test"])
        recall = calc_recall(predictions, data_dict["male"]["y_test"])
        print()
        print("Female:")
        predictions = model.predict(data_dict["female"]["X_test"])
        f1 = calc_f1(predictions, data_dict["female"]["y_test"])
        precision = calc_precision(predictions, data_dict["female"]["y_test"])
        recall = calc_recall(predictions, data_dict["female"]["y_test"])


def run_temp_map():
    group = 'low_temp'
    test_data_dict = load_test_data(load_pickle=True)
    datasets = run_seperation_2(test_data_dict)
    performance_test_data = datasets[group]  # returns dict: key-male/female, value-list of patient dicts.

    data_dict = dict()
    for map_key in performance_test_data:
        data_dict[map_key] = dict()
        performance_test_data[map_key] = get_data_rows_after_transform(performance_test_data[map_key])
        train_matrix = np.array(performance_test_data[map_key])
        data_dict[map_key]["X_test"], data_dict[map_key]["y_test"] = split_matrix(train_matrix)
        data_dict[map_key]["X_test"] = np.nan_to_num(data_dict[map_key]["X_test"])

    models = ["xgboost_0714.pkl", "adaboost_0694.pkl", "random_forest_0676.pkl"]
    for model_name in models:
        model = load_model(model_name)
        print(f"\n{model_name}")
        print("High_map:")
        predictions = model.predict(data_dict["high_map"]["X_test"])
        f1 = calc_f1(predictions, data_dict["high_map"]["y_test"])
        precision = calc_precision(predictions, data_dict["high_map"]["y_test"])
        recall = calc_recall(predictions, data_dict["high_map"]["y_test"])
        print()
        print("Low_map:")
        predictions = model.predict(data_dict["low_map"]["X_test"])
        f1 = calc_f1(predictions, data_dict["low_map"]["y_test"])
        precision = calc_precision(predictions, data_dict["low_map"]["y_test"])
        recall = calc_recall(predictions, data_dict["low_map"]["y_test"])


if __name__ == '__main__':
    # run_age_gender()
    run_temp_map()
