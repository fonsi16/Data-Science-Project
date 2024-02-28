import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, dataset_path, target):
        self.df = pd.read_csv(dataset_path)
        self.target = target

        # Validate if the target column exists in the dataset
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the dataset.")

        # Extract the labels from the target column
        self.labels = self.df[self.target]

        # Concatenate the labels to the dataset
        self.df_with_labels = pd.concat([self.df, self.labels], axis=1)

        self.valid_plot_types = ['count', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops', 'kde', 'correlation']

    def describe_variables(self):
        print("\nInformation of Data:")
        print(self.df.info())

        print("\nStatistical distribution of each variable:")
        print(self.df.describe())

    def determine_range(self):

        # Display the range of values for each variable without considering the class label
        print("\nRange of values for each variable:")
        print(self.df.max() - self.df.min())

    def age_feature(self):
        age = self.df["AgeCategory"]
        condition = [
            age == "18-24", age == "25-29",
            age == "30-34", age == "35-39",
            age == "40-44", age == "45-49",
            age == "50-54", age == "55-59",
            age == "60-64", age == "65-69",
            age == "70-74", age == "75-79",
            age == "80 or older"
        ]
        choice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.df["AgeCategory"] = np.select(condition, choice, default=0)

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
        self.df["BMI"] = np.select(condition, choice)

    def sleep_feature(self):
        sleep = self.df["SleepTime"]
        condition = [sleep < 6, sleep < 9, sleep >= 9]
        choice = [1, 2, 3]
        self.df["SleepTime"] = np.select(condition, choice)

    def race_feature(self):
        race = self.df["Race"]
        condition = [
            race == "White", race == "Black",
            race == "Hispanic", race == "Asian",
            race == "American Indian/Alaskan Native", race == "Other"]
        choice = [1, 2, 3, 4, 5, 6]
        self.df["Race"] = np.select(condition, choice)

    def genhealth_feature(self):
        genhealth = self.df["GenHealth"]
        condition = [
            genhealth == "Excellent", genhealth == "Very good",
            genhealth == "Good", genhealth == "Fair",
            genhealth == "Poor"]
        choice = [1, 2, 3, 4, 5]
        self.df["GenHealth"] = np.select(condition, choice)

    def process_data(self):

        #Turn all the features to numerical values:
        #HeartDisease (0-No / 1-Yes)
        #BMI (1-<16, 2-<17, 3-<18.5, 4-<25, 5-<30, 6-<35, 7-<40, 8->=40)
        #Smoking (0-No / 1-Yes)
        #AlcoholDrinking (0-No / 1-Yes)
        #Stroke (0-No / 1-Yes)
        #PhysicalHealth - Doesn't need it
        #MentalHealth - Doesn't need it
        #DiffWalking (0-No / 1-Yes)
        #Sex (0-Female / 1-Male)
        #AgeCategory (1-(18-24) / 2-(25-29) / 3-(30-34) / 4-(35-39) / 5-(40-44) / 6-(45-49) / 7-(50-54) / 8-(55-59) / 9-(60-64) / 10-(65-69) / 11-(70-74) / 12-(75-79) / 13-(80 or older))
        #Race (1-White / 2-Black / 3-Hispanic / 4-Asian / 5-American Indian/Alaskan Native / 6-Other)
        #Diabetic (0-No / 0-No, borderline diabetes / 2-Yes (during pregnancy) / 2-Yes)
        #PhysicalActivity (0-No / 1-Yes)
        #GenHealth - (1-Excellent / 2-Very good / 3-Good / 4-Fair / 5-Poor)
        #SleepTime (1-<6, 2-<9, 3->=9)
        #Asthma (0-No / 1-Yes)
        #KidneyDisease (0-No / 1-Yes)
        #SkinCancer (0-No / 1-Yes)

        # Process features
        self.age_feature()
        self.bmi_feature()
        #self.sleep_feature()
        self.race_feature()
        self.genhealth_feature()

        # Map categorical features to numerical values
        self.df["HeartDisease"] = self.df["HeartDisease"].map({"No": 0, "Yes": 1})
        self.df["Smoking"] = self.df["Smoking"].map({"No": 0, "Yes": 1})
        self.df["AlcoholDrinking"] = self.df["AlcoholDrinking"].map({"No": 0, "Yes": 1})
        self.df["Stroke"] = self.df["Stroke"].map({"No": 0, "Yes": 1})
        self.df["DiffWalking"] = self.df["DiffWalking"].map({"No": 0, "Yes": 1})
        self.df["Sex"] = self.df["Sex"].map({"Female": 0, "Male": 1})
        self.df["Diabetic"] = self.df["Diabetic"].map({"No": 0, "No, borderline diabetes": 0, "Yes (during pregnancy)": 1, "Yes": 1})
        self.df["PhysicalActivity"] = self.df["PhysicalActivity"].map({"No": 0, "Yes": 1})
        self.df["Asthma"] = self.df["Asthma"].map({"No": 0, "Yes": 1})
        self.df["KidneyDisease"] = self.df["KidneyDisease"].map({"No": 0, "Yes": 1})
        self.df["SkinCancer"] = self.df["SkinCancer"].map({"No": 0, "Yes": 1})

        print("\nProcessed Dataset:")
        print(self.df.info())

    def assess_quality(self):

        print("\nOriginal Dataset:")
        print(self.df.info)

        print("Missing values:\n", self.df.isnull().sum())
        print("Duplicate Rows:", self.df.duplicated().sum())

        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates(keep='first')

        #data_analysis_instance.plots(['count', 'kde'])

        print("\nDetecting outliers:")
        for feature in self.df:
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            print(f"Outliers in '{feature}':\n{outliers}" if not outliers.empty else f"No outliers in '{feature}'.")

            # Check if the feature is binary (0 or 1)
            if set(self.df[feature]) == {0, 1}:
                # Skip replacing outliers for binary features
                continue

            # Replace outliers with median value
            median_value = self.df[feature].median()
            self.df[feature] = np.where(
                (self.df[feature] < lower_bound) | (self.df[feature] > upper_bound),
                median_value,
                self.df[feature]
            )

        self.df.to_csv('data/heart_2020_cleaned.csv', encoding='utf-8', index=False)

        print("\nCleansed Dataset:")
        print(self.df.info)

    def plots(self, plot_types):
        for plot_type in plot_types:
            # Check if the selected plots are in the list of available plots
            if plot_type not in self.valid_plot_types:
                print(
                    f"Ignoring invalid plot type: {plot_type}. Supported plot types: {', '.join(self.valid_plot_types)}")
                continue

            for feature in self.df.columns:
                # Create a figure with a single subplot for each feature
                if plot_type == 'count' and feature not in ['BMI', 'SleepTime', 'PhysicalHealth', 'MenHealth', 'HeartDisease', self.target]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(x=feature, data=self.df, hue=self.target, ax=ax)
                    ax.set_title(f'Countplot of {feature} by {self.target}')
                    plt.show()
                if plot_type == 'kde' and feature in ['BMI', 'SleepTime', 'PhysicalHealth', 'MenHealth']:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    sns.kdeplot(self.df[self.df["HeartDisease"] == 1][feature], alpha=0.5, shade=True, color="red",
                                label="HeartDisease", ax=ax)
                    sns.kdeplot(self.df[self.df["HeartDisease"] == 0][feature], alpha=0.5, shade=True, color="green",
                                label="Normal", ax=ax)
                    plt.title(f'Distribution of {feature}', fontsize=18)
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    plt.show()

        if 'correlation' in plot_types:
            correlation = self.df.corr().round(2)
            plt.figure(figsize=(15, 12))
            sns.heatmap(correlation, annot=True, cmap='YlOrBr')
            plt.title('Correlation Heatmap')
            plt.show()

path = 'data/heart_2020.csv'

data_analysis_instance = DataAnalysis(path, 'HeartDisease')

data_analysis_instance.describe_variables()

# Process all the data to numeric values
data_analysis_instance.process_data()

# Determine the range of values for each variable
data_analysis_instance.determine_range()

# Verify the presence of duplicated data and remove it
data_analysis_instance.assess_quality()

# Plots after the cleansing
data_analysis_instance.plots(['count', 'correlation'])
