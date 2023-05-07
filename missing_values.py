import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
import pickle
TRAIN_PATH = r"C:\Users\einam\Downloads\data\data\train"
def load_all_patients(filename= "all_data.tsv" ,load_tsv=False):
    if load_tsv:
        df = pd.read_csv(filename, sep="\t")
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



if __name__ == '__main__':
    # df = load_all_patients(filename="all_data.parquet", load_tsv=True)
    # for col in df.columns:
    #     col_mean = df[col].mean()
    #     df[col].fillna(col_mean, inplace=True)
    # df.to_parquet("df_mean_values.parquet")

    with open('all_data_for_training.pkl', 'rb') as fp:
        all_data_for_training = pickle.load(fp)
        print("all_data_for_training loaded successfully")
        print(all_data_for_training['patient_0']["df"].shape)
