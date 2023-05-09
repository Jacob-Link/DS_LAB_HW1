import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


def top_ten_non_missing(patient_dict):
    # based on the EDA we chose to check the performance of a model based on the
    # top 10 non-missing columns which are lab results + Age and Gender
    keep_top_non_missing_cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Glucose", "Potassium", "Hct"]
    age_gender = ["Age", "Gender"]
    keep = keep_top_non_missing_cols + age_gender
    return patient_dict["df"][keep]


def chat_GPT_selection(patient_dict):
    chat_gpt_keep = ["Lactate", "WBC", "HR", "Temp", "Resp", "MAP", "O2Sat", "FiO2", "SaO2", "PaCO2", "BaseExcess",
                     "HCO3", "pH", "PTT", "Fibrinogen", "Platelets", "AST", "BUN", "Creatinine"]
    return patient_dict["df"][chat_gpt_keep]


def keep_all(patient_dict):
    return patient_dict["df"]


def drop_high_correlated(patient_dict):
    # above 0.75 and below -0.75 done during EDA
    drop_cols = ["Hct", "Bilirubin_direct", "DBP", "HCO3", "SBP", "Unit1"]
    return patient_dict["df"].drop(columns=drop_cols)


def variance_threshold():
    pass


def forward_sfs(model, X_train, y_train):
    selector = SelectFromModel(model)
    selector = selector.fit(X_train, y_train)
    status = selector.get_support()
    print("Selection status: ")
    print(status)

    # features = X_train.columns
    # print("All features:")
    # print(features)
    # print(f"Number of features prior to selection: {len(features)}")
    #
    # print("Selected features:")
    # selected_features = features[status]
    # print(selected_features)

    selected_new_x = selector.transform(X_train)
    return status, selected_new_x


def hill_climbing_draft(X_train, y_train, X_test, y_test, model):
    current_features = []
    best_f1 = 0

    while True:
        feature_candidates = [f for f in X_train.columns if f not in current_features]
        f1_scores = []

        for feature in feature_candidates:
            temp_features = current_features + [feature]
            model.fit(X_train[temp_features], y_train)
            y_pred = model.predict(X_test[temp_features])
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)

        best_candidate_f1 = np.max(f1_scores)
        best_candidate = feature_candidates[np.argmax(f1_scores)]

        if best_candidate_f1 > best_f1:
            current_features.append(best_candidate)
            best_f1 = best_candidate_f1
            print(f"Added feature '{best_candidate}', F1 score: {best_f1:.4f}")
        else:
            break

    return current_features

