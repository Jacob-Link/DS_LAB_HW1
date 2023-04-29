import pandas as pd
import matplotlib.pyplot as plt

from cleaning_data import load_data_for_eda


def get_avg_hr_per_patient(df):
    # calc avg over all values for each patient
    grouped = df.group("id")["HR"].avg()


def heart_rate_hypothesis(df):
    patient_avg_hr = get_avg_hr_per_patient(df)


if __name__ == '__main__':
    df = load_data_for_eda(load_tsv=True)
