import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, dataset_path, target):
        self.df = pd.read_csv(dataset_path)
        self.labels = self.df.columns
        self.target = target

        self.valid_plot_types = ['count', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']

    def verify_and_delete_duplicates(self):
        duplicates = self.df[self.df.duplicated(keep=False)]
        print("Duplicate Rows:", len(duplicates))
        print(duplicates)

        if len(duplicates) > 0:
            self.df = self.df.drop_duplicates(keep='first')

        self.df.to_csv('data/heart_2020_cleaned.csv', encoding='utf-8', index=False)

    def describe_variables(self):
        print("\nInformation of Data:")
        print(self.df.info())

        print("\nStatistical distribution of each variable:")
        print(self.df.describe())

    """
        1 -> severe thinness
        2 -> moderate thinness
        3 -> mild thinness
        4 -> normal
        5 -> overweight
        6 -> obese class 1
        7 -> obese class 2
        8 -> obese class 3
    """

    def bmi_feature(self):
        bmi = self.df["BMI"]
        condition = [bmi < 16, bmi < 17, bmi < 18.5, bmi < 25, bmi < 30, bmi < 35, bmi < 40, bmi >= 40]
        choice = [1, 2, 3, 4, 5, 6, 7, 8]
        self.df["BmiClass"] = np.select(condition, choice)

    def sleep_feature(self):
        sleep = self.df["SleepTime"]
        condition = [sleep < 6, sleep < 9, sleep >= 9]
        choice = [1, 2, 3]
        self.df["SleepClass"] = np.select(condition, choice)

    def count_plots(self):
        for i in range(len(self.df.columns)):
            column = self.df.columns[i]
            if column == 'BMI' or column == 'HeartDisease':
                continue
            else:
                sns.countplot(x=column, data=self.df, hue=self.target)
                plt.show()


path = 'data/heart_2020.csv'
# df = pd.read_csv(path)

data_analysis_instance = DataAnalysis(path, 'HeartDisease')

# Verify the presence of duplicated data and remove it
data_analysis_instance.verify_and_delete_duplicates()

# Create features
data_analysis_instance.bmi_feature()
data_analysis_instance.sleep_feature()

data_analysis_instance.describe_variables()

data_analysis_instance.count_plots()
