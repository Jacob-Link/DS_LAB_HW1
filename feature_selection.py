# select certain columns
# backwards or forward techniques learnt in stats 2

import pandas as pd


def top_ten_non_missing(patient_dict):
    # based on the EDA we chose to check the performance of a model based on the
    # top 10 non-missing columns which are lab results + Age and Gender
    keep_top_non_missing_cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Glucose", "Potassium", "Hct"]
    age_gender = ["Age", "Gender"]
    keep = keep_top_non_missing_cols + age_gender
    return patient_dict["df"][keep]


def keep_all(patient_dict):
    return patient_dict["df"]
