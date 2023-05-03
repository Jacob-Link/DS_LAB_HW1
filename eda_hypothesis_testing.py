import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cleaning_data import load_data_for_eda

# from scipy.stats import ttest_ind


def boxplot_two_categories(df, value_col, category_col, x_label, y_label, title):
    # create plot
    ax = plt.figure(figsize=(7, 6), facecolor=(1, 1, 1))
    ax = sns.boxplot(x=category_col, y=value_col, data=df, showmeans=True)

    # Calculate the mean value for each group
    means = df.groupby(category_col)[value_col].mean()

    # Add the mean value to the plot
    for i, mean in enumerate(means):
        ax.text(i, mean + 3, f"{mean:.2f}", horizontalalignment='center')

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")

    box_fig = ax.get_figure()
    box_fig.show()


def get_avg_hr_per_patient(df):
    # calc avg over all values for each patient
    grouped = df.groupby("id").agg({"HR": "mean", "SepsisLabel": "max"}).reset_index().rename(
        columns={"HR": "avg_hr", "SepsisLabel": "label"})
    grouped["label"] = grouped["label"].apply(lambda x: "sepsis" if x == 1 else "no sepsis")
    return grouped


def ttest_func(df, value_col, category_column, title):
    categories = list(df[category_column].unique())
    group1 = list(df.loc[df[category_column] == categories[0]][value_col].dropna())
    group2 = list(df.loc[df[category_column] == categories[1]][value_col].dropna())

    t_statistic, p_value = ttest_ind(group1, group2)

    print(f"ttest results for: {title}")
    print(f"t-statistic: {t_statistic:.10}")
    print(f"p-value: {p_value:.10}")
    print()


def top_25_percent_avg(group, col_str):
    top_25 = group[col_str].quantile(0.75)
    top_values = group[group[col_str] > top_25]
    return top_values[col_str].mean()


def low_25_percent_avg(group, col_str):
    low_25 = group[col_str].quantile(0.25)
    low_values = group[group[col_str] <= low_25]
    return low_values[col_str].mean()


def get_avg_top_25_per_patient(df, col_str, rename_col):
    result = df.groupby('id').apply(top_25_percent_avg, col_str).reset_index()
    result.columns = ['id', rename_col]
    label_df = df.groupby('id')["SepsisLabel"].max().reset_index()
    label_df.columns = ['id', 'label']
    final_res = pd.merge(result, label_df, on="id")
    return final_res


def get_avg_lower_25_per_patient(df, col_str, rename_col):
    result = df.groupby('id').apply(low_25_percent_avg, col_str).reset_index()
    result.columns = ['id', rename_col]
    label_df = df.groupby('id')["SepsisLabel"].max().reset_index()
    label_df.columns = ['id', 'label']
    final_res = pd.merge(result, label_df, on="id")
    return final_res


def heart_rate_hypothesis(df):
    # figure 1 - avg all values of HR under each patient
    patient_avg_hr = get_avg_hr_per_patient(df)
    boxplot_two_categories(patient_avg_hr, "avg_hr", "label", "Diagnosis", "Heart Rate (beats per minute)"
                           , "Average Heart Rate as a Function of Diagnosis")
    # ttest_func(patient_avg_hr, "avg_hour", "label")

    # figure 2 - avg low 25% values of HR under each patient
    patient_avg_hr_top_25 = get_avg_top_25_per_patient(df, "HR", 'Top 25% HR Avg')
    boxplot_two_categories(patient_avg_hr_top_25, "Top 25% HR Avg", "label", "Diagnosis",
                           "Heart Rate (beats per minute)"
                           , "Average Heart Rate of Top 25% of Values as a Function of Diagnosis")
    # ttest_func(patient_avg_hr_top_25, "Top 25% HR Avg", "label")


def get_age_patient(df):
    grouped = df.groupby("id").agg({"Age": "first", "SepsisLabel": "max"}).reset_index().rename(
        columns={"SepsisLabel": "label"})
    grouped["label"] = grouped["label"].apply(lambda x: "sepsis" if x == 1 else "no sepsis")
    return grouped


def get_avg_temp_patient(df):
    grouped = df.groupby("id").agg({"Temp": "mean", "SepsisLabel": "max"}).reset_index().rename(
        columns={"SepsisLabel": "label", "Temp": "avg_temp"})
    grouped["label"] = grouped["label"].apply(lambda x: "sepsis" if x == 1 else "no sepsis")
    return grouped


def get_avg_o2sat_patient(df):
    grouped = df.groupby("id").agg({"O2Sat": "mean", "SepsisLabel": "max"}).reset_index().rename(
        columns={"SepsisLabel": "label", "O2Sat": "avg_o2sat"})
    grouped["label"] = grouped["label"].apply(lambda x: "sepsis" if x == 1 else "no sepsis")
    return grouped


def get_avg_resp_patient(df):
    grouped = df.groupby("id").agg({"Resp": "mean", "SepsisLabel": "max"}).reset_index().rename(
        columns={"SepsisLabel": "label", "Resp": "avg_resp"})
    grouped["label"] = grouped["label"].apply(lambda x: "sepsis" if x == 1 else "no sepsis")
    return grouped


def age_hypothesis(df):
    patient_age = get_age_patient(df)
    boxplot_two_categories(patient_age, "Age", "label", "Diagnosis", "Age", "Patient Age as a Function of Diagnosis")
    # ttest_func(patient_age, "Age", "label")


def temp_hypothesis(df):
    # figure 1 - avg all values of temp for each patient
    patient_avg_temp = get_avg_temp_patient(df)
    boxplot_two_categories(patient_avg_temp, "avg_temp", "label", "Diagnosis", "Temperature (C)",
                           "Patient Average Temperature as a Function of Diagnosis")
    # ttest_func(patient_avg_temp, "avg_temp", "label")

    # figure 2 - avg top 25% values of temp for each patient
    patient_avg_temp_top_25 = get_avg_top_25_per_patient(df, "Temp", "avg_temp_top_25")
    boxplot_two_categories(patient_avg_temp_top_25, "avg_temp_top_25", "label", "Diagnosis", "Temperature (C)",
                           "Patient Avg Temperature of top 25% Values as a Function of Diagnosis")
    # ttest_func(patient_avg_temp_top_25, "avg_temp_top_25", "label")


def o2sat_hypothesis(df):
    # figure 1 - avg all values  for each patient
    patient_avg_o2sat = get_avg_o2sat_patient(df)
    boxplot_two_categories(patient_avg_o2sat, "avg_o2sat", "label", "Diagnosis", "O2Sat (%)",
                           "Patient Average O2Sat as a Function of Diagnosis")
    # ttest_func(patient_avg_o2sat, "avg_o2sat", "label")

    # figure 2 - avg bottom 25% values of O2Sat for each patient
    patient_avg_o2sat_bottom_25 = get_avg_lower_25_per_patient(df, "O2Sat", "avg_o2sat_low_25")
    boxplot_two_categories(patient_avg_o2sat_bottom_25, "avg_o2sat_low_25", "label", "Diagnosis", "O2Sat (%)",
                           "Patient Avg O2Sat of bottom 25% Values as a Function of Diagnosis")
    # ttest_func(patient_avg_o2sat_bottom_25, "avg_o2sat_low_25", "label")


def resp_hypothesis(df):
    # figure 1 - avg all values of respiration for each patient
    patient_avg_resp = get_avg_resp_patient(df)
    boxplot_two_categories(patient_avg_resp, "avg_resp", "label", "Diagnosis", "Respiration (breaths per minute)",
                           "Patient Average Respiration as a Function of Diagnosis")
    # ttest_func(patient_avg_resp, "avg_resp", "label")

    # figure 2 - avg top 25% values of resp for each patient
    patient_avg_resp_top_25 = get_avg_top_25_per_patient(df, "Resp", "avg_resp_top_25")
    boxplot_two_categories(patient_avg_resp_top_25, "avg_resp_top_25", "label", "Diagnosis",
                           "Respiration (breaths per minute)",
                           "Patient Avg Respiration of top 25% Values as a Function of Diagnosis")
    # ttest_func(patient_avg_resp_top_25, "avg_resp_top_25", "label")


if __name__ == '__main__':
    heart_rate = False
    age = False
    temp = False
    o2sat = False
    resp = False

    df = load_data_for_eda(load_tsv=True)
    if heart_rate:
        heart_rate_hypothesis(df)
    if age:
        age_hypothesis(df)
    if temp:
        temp_hypothesis(df)
    if o2sat:
        # the lower the value the higher the potential for sepsis
        o2sat_hypothesis(df)
    if resp:
        resp_hypothesis(df)
