import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from cleaning_data import load_data
from sklearn.metrics import f1_score

#TRAIN_PATH = r"C:\Users\Jacob Link\Desktop\Data_Science_Engineer\Year_3_Part_2\Lab in data science\HW\HW1\DS_LAB_HW1\data/train/"
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
    df = load_all_patients(filename= "all_data.tsv" ,load_tsv=True)

    print(f"there are {len(df[df['SepsisLabel'] == 1]['id'].unique())} patients with sepsis")

    # df.dropna(axis=1, how='any', inplace=True)
    df = df.fillna(0)
    X, y = df.drop(["SepsisLabel", "id"], axis=1).to_numpy(), df["SepsisLabel"].to_numpy()
    # sepsis_patients = df[df["SepsisLabel"] == 1]
    print(f"there are {len(df[df['SepsisLabel'] == 1]['id'].unique())} patients with sepsis AFTER filling nan to 0")
    clf = LogisticRegression(random_state=0,max_iter=200).fit(X, y)
    predictions = clf.predict(X)
    print(f"predictions: {predictions}\nnumber of positive predictions: {sum(predictions)}")
    # print(clf.predict_proba(X))
    print(f"Accuracy : {clf.score(X, y)}")
    print(f"F1 score: {f1_score(y, predictions)}")