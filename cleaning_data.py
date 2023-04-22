import pandas as pd
import numpy as np
import os

TRAIN_PATH = r"C:\Users\Jacob Link\Desktop\Data_Science_Engineer\Year_3_Part_2\Lab in data science\HW\HW1\DS_LAB_HW1\data/train/"


def load_data(load_tsv=False):
    if load_tsv:
        df = load_all_patients(load_tsv=True)
    else:
        dfs = load_all_patients()
        df = concat_all_files(dfs)

    return df


def load_all_patients(load_tsv=False):
    if load_tsv:
        df = pd.read_csv("all_data.tsv", sep="\t")
        print(f">>> Total of {len(df['id'].unique())} patients files loaded successfully (from tsv file)")
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
    print(f"Patients with Sepsis: {round(100 * (ones / num_patients), 2)}% [{ones}/{num_patients:,}]")
    print(
        f"Patients with Sepsis: {round(100 * ((num_patients - ones) / num_patients), 2)} [{num_patients - ones}/{num_patients:,}]")


def export(df, file_name, export=False):
    if export:
        df.to_csv(file_name, sep="\t")


if __name__ == '__main__':
    df = load_data(load_tsv=True)
    export(df, file_name="all_data.tsv", export=False)
    hours_distribution(df)
    label_balance(df)
