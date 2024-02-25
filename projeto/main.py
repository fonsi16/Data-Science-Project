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



path = 'data/heart_2020.csv'
# df = pd.read_csv(path)

data_analysis_instance = DataAnalysis(path, 'HeartDisease')

# Verify the presence of duplicated data and remove it
data_analysis_instance.verify_and_delete_duplicates()

data_analysis_instance.describe_variables()


