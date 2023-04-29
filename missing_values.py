import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
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
    df = load_all_patients(filename="all_data.tsv", load_tsv=True)
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    print(df)
    df = imputer.fit_transform(df.drop(['id'], axis=1).to_numpy())
    print(df)