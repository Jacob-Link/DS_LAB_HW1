import pandas as pd
import numpy as np
import os
import pickle

TRAIN_PATH = r"C:\Users\Jacob Link\Desktop\Data_Science_Engineer\Year_3_Part_2\Lab in data science\HW\HW1\DS_LAB_HW1\data/train/"
# TRAIN_PATH = r"C:\Users\einam\Downloads\data\train"


def modify_dfs(dfs_dict):
    # modification: remove label column, remove all rows eq 1 except 1 for those labeled 1.
    # return dict of keys: patient id,values: dict of label and df ready to input to ml model
    for patient, df in dfs_dict.items():
        df = get_relevant_rows(df)
        dfs_dict[patient] = {"label": sum(df["SepsisLabel"]), "df": df.drop(columns=["SepsisLabel"])}
    return dfs_dict


def load_train_data_for_ml_model(load_pickle=True):
    if load_pickle:
        with open('dict_ready_train.pkl', 'rb') as f:
            train_dict_dfs = pickle.load(f)
            print('>>> successfully loaded dict from pkl file')

    else:
        dict_dfs = load_all_patients_for_ml_model()
        train_dict_dfs = modify_dfs(dict_dfs)
        with open('dict_ready_train.pkl', 'wb') as f:
            pickle.dump(train_dict_dfs, f)
            print('>>> successfully saved dict as pickle')
    return train_dict_dfs


def load_data_for_eda(load_tsv=False):
    if load_tsv:
        df = load_all_patients(load_tsv=True)
    else:
        dfs = load_all_patients()
        df = concat_all_files(dfs)

    return df


def load_all_patients_for_ml_model():
    all_files = os.listdir(TRAIN_PATH)
    df_dict = dict()
    for i, f in enumerate(all_files):
        if (i + 1) % 500 == 0:
            print(f">>> loaded [{i + 1:,}/{len(all_files):,}] patients data...")
        df = pd.read_csv(TRAIN_PATH + f"/{f}", sep="|")
        id = f[:-4]
        df_dict[id] = df

    print(f">>> Total of {len(df_dict)} patients files loaded successfully")
    return df_dict


def load_all_patients(load_tsv=False):
    if load_tsv:
        df = pd.read_csv("all_data.tsv", sep="\t")
        print(f">>> Total of {len(df['id'].unique()):,} patients files loaded successfully (from tsv file)")
        return df

    else:
        all_files = os.listdir(TRAIN_PATH)
        df_list = []
        for f in all_files:
            df = pd.read_csv(TRAIN_PATH + f"/{f}", sep="|")
            df["id"] = f[:-4]  # add id column to each df
            df_list.append(df)

        print(f">>> Total of {len(df_list)} patients files loaded successfully")
        return df_list


def concat_all_files(dfs):
    df = pd.concat(dfs)
    return df


def hours_distribution(df):
    grouped = df.groupby("id").size().to_frame().rename(columns={0: "num_hours"})
    print("Description of number of hours recorded per patient: ")
    print(grouped["num_hours"].describe())


def label_balance(df):
    grouped = df.groupby("id")["SepsisLabel"].sum().reset_index().rename(columns={"SepsisLabel": "sum_label"})
    grouped["sepsis"] = grouped["sum_label"].apply(lambda x: 1 if x > 0 else 0)

    ones = sum(grouped['sepsis'])
    num_patients = len(grouped)
    print("Sepsis label balance in data:")
    print(f"Patients with Sepsis: {round(100 * (ones / num_patients), 2)}% [{ones:,}/{num_patients:,}]")
    print(
        f"Patients with Sepsis: {round(100 * ((num_patients - ones) / num_patients), 2)} [{num_patients - ones:,}/{num_patients:,}]")


def export(df, file_name, export=False):
    if export:
        df.to_csv(file_name, sep="\t", index=False)


def check_no_patient_label_only_one(df):
    check_grouped = df.groupby('id').agg({'SepsisLabel': ['sum', 'count']}).reset_index()
    check_df = check_grouped.loc[check_grouped["SepsisLabel"]["sum"] == check_grouped["SepsisLabel"]["count"]]
    if len(check_df):
        print(f"Number of patients diagnosed with Sepsis within 6 hours of entry to hospital: {len(check_df)}")
    else:
        print(f"No patient has all labels equal to 1")


def get_relevant_rows(group):
    # for each group will find the first row with label eq to 1 and return all rows including the first appearance of 1
    if group["SepsisLabel"].sum():
        group = group.reset_index(drop=True)
        return group.iloc[:group.loc[group["SepsisLabel"] == 1].index[0] + 1, :]
    else:
        return group


if __name__ == '__main__':
    eda = False
    ml_train_data = True

    if eda:
        df = load_data_for_eda(load_tsv=True)
        export(df, file_name="all_data.tsv", export=False)
        hours_distribution(df)
        label_balance(df)
        check_no_patient_label_only_one(df)

    if ml_train_data:
        train_dfs = load_train_data_for_ml_model(load_pickle=True)
