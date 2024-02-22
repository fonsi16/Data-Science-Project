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
        print("Duplicate Rows:")
        print(duplicates)

        if len(duplicates) > 0:
            self.df = self.df.drop_duplicates(keep='first')


path = 'data/heart_2020_cleaned.csv'

data_analysis_instance = DataAnalysis(path, 'HeartDisease')

data_analysis_instance.verify_and_delete_duplicates()
