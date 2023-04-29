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
        ax.text(i, mean + 2, f"{mean:.2f}", horizontalalignment='center')

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


def ttest_func(df, value_col, category_column):
    categories = list(df[category_column].unique())
    group1 = df.loc[df[category_column] == categories[0]]
    group2 = df.loc[df[category_column] == categories[1]]

    # t_statistic, p_value = ttest_ind(group1, group2)

    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.3f}")


def top_25_percent_avg(group):
    top_25 = group['HR'].quantile(0.75)
    top_values = group[group['HR'] > top_25]
    return top_values['HR'].mean()


def get_avg_top_25_hr_per_patient(df):
    result = df.groupby('id').apply(top_25_percent_avg).reset_index()
    result.columns = ['id', 'Top 25% HR Avg']
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

    # figure 2 - avg top 25% values of HR under each patient
    patient_avg_hr_top_25 = get_avg_top_25_hr_per_patient(df)
    boxplot_two_categories(patient_avg_hr_top_25, "Top 25% HR Avg", "label", "Diagnosis",
                           "Heart Rate (beats per minute)"
                           , "Average Heart Rate of Top 25% of Values as a Function of Diagnosis")
    # ttest_func(patient_avg_hr_top_25, "Top 25% HR Avg", "label")


if __name__ == '__main__':
    df = load_data_for_eda(load_tsv=True)
    heart_rate_hypothesis(df)
