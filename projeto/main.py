# %% 0- Classes
import os
import pickle
import random
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
import umap
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from plotly.figure_factory._dendrogram import sch
from pycm import ConfusionMatrix
from scipy.stats import ttest_ind, probplot, shapiro
from sklearn.base import clone
from sklearn.cluster import KMeans, OPTICS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.models import Sequential
from mlxtend.feature_selection import SequentialFeatureSelector

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Generic Class responsible for loading the dataset

    Parameters:
        filename (str): The filename of the dataset to load.
        target (str): The target of the dataset to load.

    Attributes (after loading the data):
        data (DataFrame): The main dataset containing both features and target variable.
        labels (DataFrame): The target variable.
        numerical_features (List): List of numerical features in the dataset.
        categorical_features (List): List of categorical features in the dataset.


    Methods:
        _load_data(): Loads the dataset,and assigns the data and labels to the appropriate attributes.
    """

    def __init__(self, filename, target):
        """
        Initializes the DataLoader with the filename of the dataset.

        Parameters:
            filename (str): The filename of the dataset to load.
            target (str): The target of the dataset to load.
        """
        self.filename = filename

        self.data = None
        self.target = target
        self.labels = None
        self.numerical_features = []
        self.categorical_features = []

        # Load data
        self._load_data(target)

    def _load_data(self, target):
        """
        Loads the dataset from the specified filename,
        and assigns the data and labels to the appropriate attributes.

        Parameters:
            target (str): The target of the dataset to load.
        """
        try:
            # Load the dataset
            self.data = pd.read_csv(self.filename)

            # Validate if the target column exists in the dataset
            if target not in self.data.columns:
                raise ValueError(f"Target column '{target}' not found in the dataset.")

            self.labels = self.data[target]

            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")


class DataManipulator(DataLoader):
    """
    A class for manipulating data loaded from a file.

    Parameters:
        filename (str): The path to the data file.
        target (str): The target variable in the data.

    Attributes:
        data (DataFrame): The loaded data.

    Methods:
        _describe_variables: Prints information about the data, including data info, unique values, and statistical distribution.

    Raises:
        FileNotFoundError: If the specified file is not found.

    """

    def __init__(self, filename, target):
        """
        Initialize the class with a filename and target variable.

        Parameters:
            filename (str): The path to the file.
            target (str): The name of the target variable.

        Raises:
            FileNotFoundError: If the file is not found.

        """
        try:
            super().__init__(filename, target)
            print("\nData Description:")
            self._describe_variables()
        except FileNotFoundError:
            print("File not found. Please check the file path.")

    def _describe_variables(self):
        """
        Prints information about the data, including data info, unique values, and statistical distribution.
        """
        print("\nInformation of Data:")
        print(self.data.info())

        print("\nUnique values of features:")
        print(self.data.nunique())

        print("\nStatistical distribution of each variable:")
        print(self.data.describe())

    def update_data(self, filename):
        """
        Updates the data attribute with the data from the specified file.

        Parameters:
            filename (str): The path to the file.

        Raises:
            FileNotFoundError: If the file is not found.

        """
        try:
            self.data = pd.read_csv(filename)
            print("Data updated successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file path.")


class DataPreProcessing:
    """
    Class for performing data preprocessing tasks, mostly encoding.

    Parameters:
        data_loader (DataLoader): The DataLoader object containing the dataset.

    Attributes:
        data_loader (DataLoader): The DataLoader object containing the dataset.

    Methods:
        _sanity_check(): Performs a sanity check on the DataLoader object.
        _determine_range(): Displays the range of values for each variable without considering the class label.
        _age_encode(): Encodes the AgeCategory variable into numerical values.
        _encode_numerical_values(column, mapping): Encodes a variable into numerical values using the provided mapping.
        _encode_data(): Encodes categorical features into numerical values and fills numerical and categorical features arrays.
    """

    def __init__(self, data_loader):
        """
        Initializes an instance of the class.

        Parameters:
            data_loader: The data loader object used to load the data.
        """
        self.data_loader = data_loader

        self._sanity_check()

        self._encode_data()

        self._determine_range()

    def _sanity_check(self):
        """
        Performs a sanity check on the DataLoader object.

        Raises:
            ValueError: If the DataLoader object is not provided or is not a pandas DataFrame.
        """
        try:
            if not self.data_loader:
                raise ValueError("DataLoader object is not provided.")
            if not isinstance(self.data_loader.data, pd.DataFrame):
                raise ValueError("Invalid DataLoader object. It should contain a pandas DataFrame.")
        except Exception as error:
            print(f"Error occurred: {error}")
            return False

    def _determine_range(self):
        """
        Displays the range of values for each variable without considering the class label.
        """
        print("\nRange of values for each variable:")
        print(self.data_loader.data.drop(columns=["HeartDisease"]).max() - self.data_loader.data.drop(
            columns=["HeartDisease"]).min())

    def _age_encode(self):
        """
        Encodes AgeCategory into numerical values.
        """
        age_map = {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4,
            "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
            "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12,
            "80 or older": 13
        }
        self.data_loader.data["AgeCategory"] = self.data_loader.data["AgeCategory"].map(age_map)

    def _encode_numerical_values(self, column, mapping):
        """
        Encodes a variable into numerical values using the provided mapping.

        Parameters:
            column (str): The name of the column to be encoded.
            mapping (dict): The mapping of categorical values to numerical values.
        """
        self.data_loader.data[column] = self.data_loader.data[column].map(mapping)

    def _encode_data(self):
        """
        Encodes categorical features into numerical values and fills numerical and categorical features arrays.
        """
        # Map categorical features to numerical values
        categorical_mappings = {
            "HeartDisease": {"No": 0, "Yes": 1},
            "Smoking": {"No": 0, "Yes": 1},
            "AlcoholDrinking": {"No": 0, "Yes": 1},
            "Stroke": {"No": 0, "Yes": 1},
            "DiffWalking": {"No": 0, "Yes": 1},
            "Sex": {"Female": 0, "Male": 1},
            "Diabetic": {"No": 0, "No, borderline diabetes": 0, "Yes (during pregnancy)": 1, "Yes": 1},
            "PhysicalActivity": {"No": 0, "Yes": 1},
            "Asthma": {"No": 0, "Yes": 1},
            "KidneyDisease": {"No": 0, "Yes": 1},
            "SkinCancer": {"No": 0, "Yes": 1}
        }
        for column, mapping in categorical_mappings.items():
            self._encode_numerical_values(column, mapping)

        # Encode AgeCategory, Race, and GenHealth
        self._age_encode()
        self._encode_numerical_values("Race", {"White": 1, "Black": 2, "Hispanic": 3, "Asian": 4,
                                               "American Indian/Alaskan Native": 5, "Other": 6})
        self._encode_numerical_values("GenHealth", {"Excellent": 5, "Very good": 4, "Good": 3, "Fair": 2, "Poor": 1})

        # Fill the numerical and categorical features arrays
        self.data_loader.categorical_features.append("HeartDisease")
        self.data_loader.categorical_features.append("Smoking")
        self.data_loader.categorical_features.append("AlcoholDrinking")
        self.data_loader.categorical_features.append("Stroke")
        self.data_loader.categorical_features.append("DiffWalking")
        self.data_loader.categorical_features.append("Sex")
        self.data_loader.categorical_features.append("Race")
        self.data_loader.categorical_features.append("Diabetic")
        self.data_loader.categorical_features.append("PhysicalActivity")
        self.data_loader.categorical_features.append("GenHealth")
        self.data_loader.categorical_features.append("Asthma")
        self.data_loader.categorical_features.append("KidneyDisease")
        self.data_loader.categorical_features.append("SkinCancer")

        self.data_loader.numerical_features.append("BMI")
        self.data_loader.numerical_features.append("PhysicalHealth")
        self.data_loader.numerical_features.append("MentalHealth")
        self.data_loader.numerical_features.append("AgeCategory")
        self.data_loader.numerical_features.append("SleepTime")

        print("\nProcessed Dataset:")
        print(self.data_loader.data.info())


class DataCleaning:
    """
    A class for performing data cleaning operations on a dataset.

    Parameters:
        data_loader (DataLoader): An instance of the DataLoader class that provides access to the dataset.

    Attributes:
        data_loader (DataLoader): An instance of the DataLoader class that provides access to the dataset.

    Methods:
        handle_missing_values: Removes rows with missing values from the dataset.
        remove_duplicates: Removes duplicate rows from the dataset.
        detect_and_remove_outliers: Detects and removes outliers from the dataset.
    """

    def __init__(self, data_loader):
        """
        Initializes an instance of the class.

        Parameters:
            data_loader: The data loader object used to load the dataset.

        Returns:
            None
        """
        self.data_loader = data_loader
        print("\nOriginal Dataset before cleaning:")
        print(self.data_loader.data.info())

    def handle_missing_values(self):
        """
        This method checks for missing values in the dataset and removes any rows that contain missing values.
        It prints the number of missing values for each column before and after removing the rows.
        If there are no missing values, no rows are removed.
        """
        print("Missing values:\n", self.data_loader.data.isnull().sum())

        if self.data_loader.data.isnull().sum().sum() > 0:
            self.data_loader.data = self.data_loader.data.dropna()

    def remove_duplicates(self):
        """
        This method checks for duplicate rows in the dataset and removes them if any are found.
        It prints the number of duplicate rows before and after the removal process.
        """
        print("Duplicate Rows:", self.data_loader.data.duplicated().sum())

        if self.data_loader.data.duplicated().sum() > 0:
            self.data_loader.data = self.data_loader.data.drop_duplicates(keep='first')

    def detect_and_remove_outliers(self):
        """
        This method iterates over each feature in the dataset and detects outliers using the interquartile range (IQR) method.
        Outliers are defined as values that fall below the lower bound (Q1 - 1.5 * IQR) or above the upper bound (Q3 + 1.5 * IQR).
        Outliers are then removed from the dataset.

        If a feature has only two unique values, it is skipped as it is not suitable for outlier detection.
        After removing outliers, if a feature has only one unique value, it is considered redundant and is deleted from the dataset.
        """
        print("\nDetecting outliers:")
        features_to_delete = []
        for feature in self.data_loader.data.columns:
            # Skip features with only two unique values
            if len(self.data_loader.data[feature].unique()) == 2:
                print(f"Skipping '{feature}' as it has only two unique values.")
                continue

            q1 = self.data_loader.data[feature].quantile(0.25)
            q3 = self.data_loader.data[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_indices = self.data_loader.data[
                (self.data_loader.data[feature] < lower_bound) | (self.data_loader.data[feature] > upper_bound)].index

            print(f"Outliers in '{feature}'." if not outliers_indices.empty else f"No outliers in '{feature}'.")

            self.data_loader.data.drop(outliers_indices, inplace=True)

            # Verify if the feature after removing outliers has only one unique value
            if len(self.data_loader.data[feature].unique()) == 1:
                print(f"Feature '{feature}' has only one unique value after removing outliers. Deleting it.")
                features_to_delete.append(feature)

        # Remove features with only one unique value
        self.data_loader.data.drop(columns=features_to_delete, inplace=True)


class DataVisualization:
    """
    A class for visualizing data using various plot types.
    
    Parameters:
        data_loader (DataLoader): A DataLoader object that provides access to the data.
        valid_plot_types (list): A list of valid plot types that can be used for visualization.
    
    Attributes:
        data_loader (DataLoader): A DataLoader object that provides access to the data.
        valid_plot_types (list): A list of valid plot types that can be used for visualization.
        labels (list): A list of unique labels in the dataset.
    
    Methods:
        plot_all_features(): Plots histograms for all features in the dataset.
        plots(plot_types): Plots the specified types of plots for each feature in the dataset.
    """

    def __init__(self, data_loader, valid_plot_types):
        """
        Initializes the DataVisualization class with a DataLoader object.

        Parameters:
        - data_loader (DataLoader): The DataLoader object used to load the data.
        - valid_plot_types (list): A list of valid plot types that can be used for visualization.

        Attributes:
        - data_loader (DataLoader): The DataLoader object used to load the data.
        - valid_plot_types (list): A list of valid plot types that can be used for visualization.
        - labels (list): A list of unique labels in the loaded data.

        """
        self.data_loader = data_loader
        self.valid_plot_types = valid_plot_types
        self.labels = self.data_loader.data[self.data_loader.target].unique().tolist()

    def plot_all_features(self):
        """
        Plots histograms for all features in the dataset.

        This method generates a histogram for each feature in the dataset. The histograms show the frequency distribution
        of values for each feature. If labels are provided, multiple histograms will be plotted for each feature, one for
        each label.
        """
        num_features = len(self.data_loader.data.columns.tolist())
        num_cols = 3  # Adjust the number of columns to control subplot arrangement
        num_rows = int(np.ceil(num_features / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

        for idx, ax in enumerate(axes.flat):
            if idx < num_features:
                ax.set_title(f'Feature {self.data_loader.data.columns.tolist()[idx]}', fontsize=12)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True)

                if self.labels is not None:
                    # Add a plot per feature and label
                    for label in self.labels:
                        mask = np.array(self.data_loader.data[self.data_loader.target] == label)
                        ax.hist(self.data_loader.data.loc[mask, self.data_loader.data.columns.tolist()[idx]], bins=20,
                                alpha=0.7, label=label)
                    ax.legend()

        plt.tight_layout()
        plt.show()

    def plots(self, plot_types):
        """
        Plots the specified types of plots for each feature in the dataset.
        
        Parameters:
        - plot_types (list): A list of plot types to be plotted.
        """
        for plot_type in plot_types:
            # Check if the selected plots are in the list of available plots
            if plot_type not in self.valid_plot_types:
                print(
                    f"Ignoring invalid plot type: {plot_type}. Supported plot types: {', '.join(self.valid_plot_types)}")
                continue

            for feature in self.data_loader.data.columns:
                # Create a figure with a single subplot for each feature
                if plot_type == 'box' and self.data_loader.data[feature].nunique() > 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=self.data_loader.target, y=feature, data=self.data_loader.data, ax=ax)
                    ax.set_title(f'Boxplot of {feature} by {self.data_loader.target}')

                    # Set the tick labels on the x-axis to "No" and "Yes"
                    ax.set_xticklabels(['No', 'Yes'])

                    plt.show()

        if 'correlation' in plot_types:
            correlation = self.data_loader.data.corr().round(2)
            heartdisease_correlation = correlation['HeartDisease'].sort_values(ascending=False)

            plt.figure(figsize=(15, 12))
            sns.heatmap(correlation, annot=True, cmap='YlOrBr', annot_kws={'size': 8})
            plt.title('Correlation Heatmap')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print("\nCorrelation of the features with Heart Disease:\n")
            print(heartdisease_correlation)

        if 'barh' in plot_types:
            # Train a RandomForestClassifier model
            clf = RandomForestClassifier()
            X = self.data_loader.data.drop(columns=[self.data_loader.target])  # Features
            y = self.data_loader.data[self.data_loader.target]  # Target variable
            clf.fit(X, y)

            # Calculate permutation importance
            result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            perm_sorted_idx = result.importances_mean.Parametersort()

            # Visualize feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x=result.importances_mean[perm_sorted_idx], y=X.columns[perm_sorted_idx], color='blue')
            plt.xlabel('Permutation Importance')
            plt.ylabel('Features')
            plt.title('Permutation Importance')
            plt.show()


class DimensionalityReduction:
    """
    Class for performing dimensionality reduction techniques such as PCA and UMAP.

    Parameters:
        data_loader (DataLoader): An instance of the DataLoader class that provides the data.

    Attributes:
        data_loader (DataLoader): An instance of the DataLoader class that provides the data.
        dataset (DataFrame): A sample of 30% of the data.
        data (nparray): The standardized data.
        target (Series): The target variable from the data.

    Methods:
        plot_projection(projection, title): Plot the projection of the data.
        compute_pca(n_components): Perform Principal Component Analysis (PCA) on the data.
        compute_umap(n_components, n_neighbors, min_dist, metric): Perform Uniform Manifold Approximation and Projection (UMAP) on the data.
    """

    def __init__(self, data_loader):
        """
        Initializes an instance of MyClass.

        Parameters:
        - data_loader (DataLoader): An object that loads the data.

        Attributes:
        - data_loader: The data loader object.
        - dataset: A sample of 30% of the data.
        - data: The standardized data.
        - target: The target variable from the data.
        """
        self.data_loader = data_loader

        # Sample 30% of the data
        self.dataset = self.data_loader.data.sample(frac=0.3, random_state=42)

        self.data = StandardScaler().fit_transform(self.data_loader.data.drop(columns=[self.data_loader.target]))
        self.target = self.data_loader.data[self.data_loader.target]

    def plot_projection(self, projection, title):
        """
        Plot the projection of the data.

        Parameters:
        - projection: The projected data.
        - title (str): The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        if projection.shape[1] == 1:
            plt.scatter(projection, np.zeros_like(projection), c=self.target, alpha=0.5, cmap='viridis')
        else:
            plt.scatter(projection[:, 0], projection[:, 1], c=self.target, alpha=0.5, cmap='viridis')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

    def compute_pca(self, n_components=2):
        """
        Perform Principal Component Analysis (PCA) on the data.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - The projected data after PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Perform Uniform Manifold Approximation and Projection (UMAP) on the data.

        Parameters:
        - n_components: The number of components to keep.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between points in the low-dimensional representation.
        - metric: The distance metric to use.

        Returns:
        - The projected data after UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)


class HypothesisTester:
    """
    Class for performing hypothesis tests and generating Q-Q plots.

    Parameters:
    - data_loader: An instance of the DataLoader class used for loading the data.

    Attributes:
    - data_loader: An instance of the DataLoader class used for loading the data.
    - BMI_with_HD: Column data for BMI with Heart Disease.
    - Smoke_with_HD: Column data for Smoking with Heart Disease.
    - Alcohol_with_HD: Column data for Alcohol Drinking with Heart Disease.
    - Stroke_with_HD: Column data for Stroke with Heart Disease.
    - PH_with_HD: Column data for Physical Health with Heart Disease.
    - MH_with_HD: Column data for Mental Health with Heart Disease.
    - DW_with_HD: Column data for Diff Walking with Heart Disease.
    - Sex_with_HD: Column data for Sex with Heart Disease.
    - AC_with_HD: Column data for Age Category with Heart Disease.
    - Diabetic_with_HD: Column data for Diabetic with Heart Disease.
    - PA_with_HD: Column data for Physical Activity with Heart Disease.
    - GH_with_HD: Column data for Gen Health with Heart Disease.
    - ST_with_HD: Column data for Sleep Time with Heart Disease.
    - Asthma_with_HD: Column data for Asthma with Heart Disease.
    - KD_with_HD: Column data for Kidney Disease with Heart Disease.
    - SC_with_HD: Column data for Skin Cancer with Heart Disease.
    - BMI_without_HD: Column data for BMI without Heart Disease.
    - Smoke_without_HD: Column data for Smoking without Heart Disease.
    - Alcohol_without_HD: Column data for Alcohol Drinking without Heart Disease.
    - Stroke_without_HD: Column data for Stroke without Heart Disease.
    - PH_without_HD: Column data for Physical Health without Heart Disease.
    - MH_without_HD: Column data for Mental Health without Heart Disease.
    - DW_without_HD: Column data for Diff Walking without Heart Disease.
    - Sex_without_HD: Column data for Sex without Heart Disease.
    - AC_without_HD: Column data for Age Category without Heart Disease.
    - Diabetic_without_HD: Column data for Diabetic without Heart Disease.
    - PA_without_HD: Column data for Physical Activity without Heart Disease.
    - GH_without_HD: Column data for Gen Health without Heart Disease.
    - ST_without_HD: Column data for Sleep Time without Heart Disease.
    - Asthma_without_HD: Column data for Asthma without Heart Disease.
    - KD_without_HD: Column data for Kidney Disease without Heart Disease.
    - SC_without_HD: Column data for Skin Cancer without Heart Disease.
    - variable_names: List of variable names.
    - data_samples: Tuple of data samples.
    - normal_distributed_variables_with_HD: List of normal distributed variables with Heart Disease.
    - normal_distributed_variables_without_HD: List of normal distributed variables without Heart Disease.
    - not_normal_distributed_variables_with_HD: List of not normal distributed variables with Heart Disease.
    - not_normal_distributed_variables_without_HD: List of not normal distributed variables without Heart Disease.

    Methods:
    - _wilcoxon_ranksum_test(self, group1, group2): Perform Wilcoxon rank-sum test (Mann-Whitney U test) for two independent samples.
    - _unpaired_t_test(self, group1, group2): Perform unpaired t-test for two independent samples.
    - perform_tests(self): Perform hypothesis tests for all variable pairs.
    - qq_plots(self, distribution='norm'): Generate Q-Q plots for all variables.
    - test_normality(self): Test the normality assumption for all variables.
    - distribute_normality_data(self): Distribute data based on normality assumption.
    """

    def __init__(self, data_loader):
        """
        Initialize the HypothesisTester object.

        Parameters:
        - data_loader: An instance of the DataLoader class used for loading the data.
        """
        self.data_loader = data_loader

        # Column Data with Hearth Disease
        self.BMI_with_HD = self.data_loader.data['BMI'][self.data_loader.data['HeartDisease'] == 1]
        self.Smoke_with_HD = self.data_loader.data['Smoking'][self.data_loader.data['HeartDisease'] == 1]
        self.Alcohol_with_HD = self.data_loader.data['AlcoholDrinking'][self.data_loader.data['HeartDisease'] == 1]
        self.Stroke_with_HD = self.data_loader.data['Stroke'][self.data_loader.data['HeartDisease'] == 1]
        self.PH_with_HD = self.data_loader.data['PhysicalHealth'][self.data_loader.data['HeartDisease'] == 1]
        self.MH_with_HD = self.data_loader.data['MentalHealth'][self.data_loader.data['HeartDisease'] == 1]
        self.DW_with_HD = self.data_loader.data['DiffWalking'][self.data_loader.data['HeartDisease'] == 1]
        self.Sex_with_HD = self.data_loader.data['Sex'][self.data_loader.data['HeartDisease'] == 1]
        self.AC_with_HD = self.data_loader.data['AgeCategory'][self.data_loader.data['HeartDisease'] == 1]
        self.Diabetic_with_HD = self.data_loader.data['Diabetic'][self.data_loader.data['HeartDisease'] == 1]
        self.PA_with_HD = self.data_loader.data['PhysicalActivity'][self.data_loader.data['HeartDisease'] == 1]
        self.GH_with_HD = self.data_loader.data['GenHealth'][self.data_loader.data['HeartDisease'] == 1]
        self.ST_with_HD = self.data_loader.data['SleepTime'][self.data_loader.data['HeartDisease'] == 1]
        self.Asthma_with_HD = self.data_loader.data['Asthma'][self.data_loader.data['HeartDisease'] == 1]
        self.KD_with_HD = self.data_loader.data['KidneyDisease'][self.data_loader.data['HeartDisease'] == 1]
        self.SC_with_HD = self.data_loader.data['SkinCancer'][self.data_loader.data['HeartDisease'] == 1]

        # Column Data without Hearth Disease
        self.BMI_without_HD = self.data_loader.data['BMI'][self.data_loader.data['HeartDisease'] == 0]
        self.Smoke_without_HD = self.data_loader.data['Smoking'][self.data_loader.data['HeartDisease'] == 0]
        self.Alcohol_without_HD = self.data_loader.data['AlcoholDrinking'][self.data_loader.data['HeartDisease'] == 0]
        self.Stroke_without_HD = self.data_loader.data['Stroke'][self.data_loader.data['HeartDisease'] == 0]
        self.PH_without_HD = self.data_loader.data['PhysicalHealth'][self.data_loader.data['HeartDisease'] == 0]
        self.MH_without_HD = self.data_loader.data['MentalHealth'][self.data_loader.data['HeartDisease'] == 0]
        self.DW_without_HD = self.data_loader.data['DiffWalking'][self.data_loader.data['HeartDisease'] == 0]
        self.Sex_without_HD = self.data_loader.data['Sex'][self.data_loader.data['HeartDisease'] == 0]
        self.AC_without_HD = self.data_loader.data['AgeCategory'][self.data_loader.data['HeartDisease'] == 0]
        self.Diabetic_without_HD = self.data_loader.data['Diabetic'][self.data_loader.data['HeartDisease'] == 0]
        self.PA_without_HD = self.data_loader.data['PhysicalActivity'][self.data_loader.data['HeartDisease'] == 0]
        self.GH_without_HD = self.data_loader.data['GenHealth'][self.data_loader.data['HeartDisease'] == 0]
        self.ST_without_HD = self.data_loader.data['SleepTime'][self.data_loader.data['HeartDisease'] == 0]
        self.Asthma_without_HD = self.data_loader.data['Asthma'][self.data_loader.data['HeartDisease'] == 0]
        self.KD_without_HD = self.data_loader.data['KidneyDisease'][self.data_loader.data['HeartDisease'] == 0]
        self.SC_without_HD = self.data_loader.data['SkinCancer'][self.data_loader.data['HeartDisease'] == 0]

        self.variable_names = ['BMI_with_HD', 'Smoke_with_HD', 'Alcohol_with_HD', 'Stroke_with_HD', 'PH_with_HD',
                               'MH_with_HD', 'DW_with_HD',
                               'Sex_with_HD', 'AC_with_HD', 'Diabetic_with_HD', 'PA_with_HD', 'GH_with_HD',
                               'ST_with_HD',
                               'Asthma_with_HD', 'KD_with_HD', 'SC_with_HD', 'BMI_without_HD', 'Smoke_without_HD',
                               'Alcohol_without_HD',
                               'Stroke_without_HD', 'PH_without_HD', 'MH_without_HD', 'DW_without_HD', 'Sex_without_HD',
                               'AC_without_HD',
                               'Diabetic_without_HD', 'PA_without_HD', 'GH_without_HD', 'ST_without_HD',
                               'Asthma_without_HD', 'KD_without_HD', 'SC_without_HD']
        self.data_samples = (self.BMI_with_HD, self.Smoke_with_HD, self.Alcohol_with_HD,
                             self.Stroke_with_HD, self.PH_with_HD, self.MH_with_HD, self.DW_with_HD, self.Sex_with_HD,
                             self.AC_with_HD, self.Diabetic_with_HD, self.PA_with_HD, self.GH_with_HD, self.ST_with_HD,
                             self.Asthma_with_HD, self.KD_with_HD, self.SC_with_HD, self.BMI_without_HD,
                             self.Smoke_without_HD, self.Alcohol_without_HD, self.Stroke_without_HD,
                             self.PH_without_HD, self.MH_without_HD, self.DW_without_HD, self.Sex_without_HD,
                             self.AC_without_HD, self.Diabetic_without_HD, self.PA_without_HD, self.GH_without_HD,
                             self.ST_without_HD, self.Asthma_without_HD, self.KD_without_HD, self.SC_without_HD)

        self.normal_distributed_variables_with_HD = []

        self.normal_distributed_variables_without_HD = []

        self.not_normal_distributed_variables_with_HD = []

        self.not_normal_distributed_variables_without_HD = []

    def _wilcoxon_ranksum_test(self, group1, group2):
        """
        Perform Wilcoxon rank-sum test (Mann-Whitney U test) for two independent samples.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = sms.stattools.stats.mannwhitneyu(group1, group2)

        return statistic, p_value

    def _unpaired_t_test(self, group1, group2):
        """
        Perform unpaired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_ind(group1, group2)
        return t_statistic, p_value

    def perform_tests(self):
        """
        Perform hypothesis tests for the normal and not normal distributed variables.

        Prints the results of the tests.
        """
        print("\nUnpaired T-test tests for the normal distributed variables:")
        # Iterate over the indices of the arrays of the normal distributed variables
        for i in range(len(self.normal_distributed_variables_with_HD)):
            # Perform Unpaired T-Test
            t_stat, p_val = self._unpaired_t_test(self.normal_distributed_variables_with_HD[i],
                                                  self.normal_distributed_variables_without_HD[i])

            # Print the results
            print(f"\nUnpaired T-test test between the array of "
                  f"{self.normal_distributed_variables_with_HD[i].name} with HeartDisease and the array without : ")
            print("t-statistic:", t_stat)
            print("p-value:", p_val)

        print("\nWilcoxon rank-sum tests for the not normal distributed variables:")
        # Iterate over the indices of the arrays of the not normal distributed variables
        for i in range(len(self.not_normal_distributed_variables_with_HD)):
            # Perform Wilcoxon rank-sum test
            statistic, p_value = self._wilcoxon_ranksum_test(self.not_normal_distributed_variables_with_HD[i],
                                                             self.not_normal_distributed_variables_without_HD[i])

            # Print the results
            print(f"\nWilcoxon rank-sum test between the array of "
                  f"{self.not_normal_distributed_variables_with_HD[i].name} with HeartDisease and the array without : ")
            print("Test statistic:", statistic)
            print("p-value:", p_value)

    def qq_plots(self, distribution='norm'):
        """
        Generate Q-Q plots for multiple data samples.

        Parameters:
        - distribution: String indicating the theoretical distribution to compare against. Default is 'norm' for normal
        distribution.
        """
        num_samples = len(self.data_samples)
        num_rows = (num_samples + 1) // 2  # Calculate the number of rows for subplots
        num_cols = 2 if num_samples > 1 else 1  # Ensure at least 1 column for subplots

        # Adjust the height of the figure to fit all Q-Q plots without overlapping
        fig_height = 6 * num_rows  # Adjust this value as needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, fig_height))
        axes = axes.flatten()  # Flatten axes if multiple subplots

        for i, data in enumerate(self.data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f'Q-Q Plot ({distribution})')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel(self.variable_names[i])

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()

    def test_normality(self):
        """
        Test the normality of multiple data samples using Shapiro-Wilk test.

        Parameters:
        - variable_names: List with the names of the variables to be tested.
        - data_samples: Variable number of 1D array-like objects representing the data samples.

        Returns:
        - results: Dictionary containing the test results for each data sample.
                   The keys are the variable names and the values are a tuple (test_statistic, p_value) for
                   Shapiro-Wilk test.
        """

        print('\nNormality Test:\n')

        results = {}
        normality = []
        for name, data in zip(self.variable_names, self.data_samples):
            results[name] = shapiro(data)
        for variable_name, shapiro_result in results.items():
            print(f'{variable_name}:')
            print(f'Shapiro-Wilk test - Test statistic: {shapiro_result.statistic}, p-value: {shapiro_result.pvalue}')

            if shapiro_result.pvalue > 0.05:
                normality.append(variable_name)

        if normality:
            print("\nThis variables seem normally distributed:", normality)
        else:
            print("\nNo variable seems normally distributed.")

    def distribute_normality_data(self):
        """
        Distributes the data samples into different lists based on their normality.

        This method iterates over the variable names and data samples, and categorizes them into different lists
        based on their normality. If the variable name is 'BMI_with_HD', the data sample is added to the
        'normal_distributed_variables_with_HD' list. If the variable name is 'BMI_without_HD', the data sample is
        added to the 'normal_distributed_variables_without_HD' list. If the variable name contains 'with_HD', the
        data sample is added to the 'not_normal_distributed_variables_with_HD' list. If the variable name contains
        'without_HD', the data sample is added to the 'not_normal_distributed_variables_without_HD' list.
        """
        for variable_name, data_sample in zip(self.variable_names, self.data_samples):
            if variable_name == 'BMI_with_HD':
                self.normal_distributed_variables_with_HD.append(data_sample)
            elif variable_name == 'BMI_without_HD':
                self.normal_distributed_variables_without_HD.append(data_sample)
            else:
                if "with_HD" in variable_name:
                    self.not_normal_distributed_variables_with_HD.append(data_sample)
                if "without_HD" in variable_name:
                    self.not_normal_distributed_variables_without_HD.append(data_sample)


class FeatureCreation:
    """
    A class that contains methods to create various features based on the data_loader object.

    Parameters:
        data_loader (object): An object that loads the data.

    Attributes:
        data_loader (object): An object that loads the data.

    Methods:
        _bmi_class_feature: Creates a BMI class feature based on the BMI column of the data.
        _sleep_class_feature: Creates a sleep class feature based on the SleepTime column of the data.
        _bad_habits_score_feature: Creates a bad habits score feature based on the Smoking and AlcoholDrinking columns of the data.
        _diseases_feature: Creates a diseases feature based on the KidneyDisease, Asthma, SkinCancer, and Diabetic columns of the data.
        _poor_health_days_month: Creates a poor health days per month feature based on the MentalHealth and PhysicalHealth columns of the data.
        _dangerous_age_stroke: Creates a dangerous stroke feature based on the Stroke and AgeCategory columns of the data.
        _age_bmi_interaction_feature: Creates an age-BMI interaction feature based on the AgeCategory and BMI columns of the data.
        _bmi_sleep_interaction_feature: Creates a BMI-sleep interaction feature based on the BMI and SleepTime columns of the data.
        _age_gh_interaction_feature: Creates an age-general health interaction feature based on the AgeCategory and GenHealth columns of the data.
        _age_sleep_interaction_feature: Creates an age-sleep interaction feature based on the AgeCategory and SleepTime columns of the data.
        create_modified_features: Calls the _bmi_class_feature and _sleep_class_feature methods to create modified features.
        create_joined_features: Calls the _bad_habits_score_feature, _diseases_feature, _poor_health_days_month, and _dangerous_age_stroke methods to create joined features.
        create_interaction_features: Calls the _age_bmi_interaction_feature, _bmi_sleep_interaction_feature, _age_gh_interaction_feature, and _age_sleep_interaction_feature methods to create interaction features.
    """

    def __init__(self, data_loader):
        """
        Initialize the class with a data loader.

        Parameters:
            data_loader: The data loader object used to load data.
        """
        self.data_loader = data_loader

    def _bmi_class_feature(self):
        """
        Creates a BMI class feature based on the BMI column of the data.
        The BMI class is determined by the following conditions:
        - BMI < 18.5: Class 1
        - 18.5 <= BMI < 25: Class 2
        - 25 <= BMI < 30: Class 3
        - 30 <= BMI < 35: Class 4
        - 35 <= BMI < 40: Class 5
        - BMI >= 40: Class 6
        The BMI class is stored in the "BMIClass" column of the data_loader object.
        """
        bmi = self.data_loader.data["BMI"]
        condition = [bmi < 18.5, bmi < 25, bmi < 30, bmi < 35, bmi < 40, bmi >= 40]
        choice = [1, 2, 3, 4, 5, 6]
        self.data_loader.data["BMIClass"] = np.select(condition, choice)
        print("Created BMIClass feature\n")

    def _sleep_class_feature(self):
        """
        Creates a sleep class feature based on the SleepTime column of the data.
        The sleep class is determined by the following conditions:
        - SleepTime < 6: Class 1
        - 6 <= SleepTime < 9: Class 2
        - SleepTime >= 9: Class 3
        The sleep class is stored in the "SleepClass" column of the data_loader object.
        """
        sleep = self.data_loader.data["SleepTime"]
        condition = [sleep < 6, sleep < 9, sleep >= 9]
        choice = [1, 2, 3]
        self.data_loader.data["SleepClass"] = np.select(condition, choice)
        print("Created SleepClass feature\n")

    def _bad_habits_score_feature(self):
        """
        Creates a bad habits score feature based on the Smoking and AlcoholDrinking columns of the data.
        The bad habits score is calculated by summing the values of the Smoking and AlcoholDrinking columns.
        The bad habits score is stored in the "BadHabitsScore" column of the data_loader object.
        """
        smoker = self.data_loader.data["Smoking"]
        alcohol = self.data_loader.data["AlcoholDrinking"]
        condition = (smoker + alcohol)
        self.data_loader.data["BadHabitsScore"] = condition
        print("Created BadHabitsScore feature\n")

    def _diseases_feature(self):
        """
        Creates a diseases feature based on the KidneyDisease, Asthma, SkinCancer, and Diabetic columns of the data.
        The diseases feature is calculated by summing the values of the KidneyDisease, Asthma, SkinCancer, and Diabetic columns.
        The diseases feature is stored in the "Diseases" column of the data_loader object.
        """
        kidney_disease = self.data_loader.data["KidneyDisease"]
        asthma = self.data_loader.data["Asthma"]
        skin_cancer = self.data_loader.data["SkinCancer"]
        diabetic = self.data_loader.data["Diabetic"]
        condition = (kidney_disease + asthma + skin_cancer + diabetic)
        self.data_loader.data["Diseases"] = condition
        print("Created Diseases feature\n")

    def _poor_health_days_month(self):
        """
        Creates a poor health days per month feature based on the MentalHealth and PhysicalHealth columns of the data.
        The poor health days per month is calculated by summing the values of the MentalHealth and PhysicalHealth columns and dividing by 30.
        The poor health days per month feature is stored in the "PoorHealthDaysMonth" column of the data_loader object.
        """
        mental_health = self.data_loader.data["MentalHealth"]
        physical_health = self.data_loader.data["PhysicalHealth"]
        condition = (mental_health + physical_health) / 30
        self.data_loader.data["PoorHealthDaysMonth"] = condition
        print("Created PoorHealthDaysMonth feature\n")

    def _dangerous_age_stroke(self):
        """
        Creates a dangerous stroke feature based on the Stroke and AgeCategory columns of the data.
        The dangerous stroke feature is determined by the following conditions:
        - AgeCategory >= 10 and Stroke = 1: 1
        - Otherwise: 0
        The dangerous stroke feature is stored in the "DangerousStroke" column of the data_loader object.
        """
        strokes = self.data_loader.data["Stroke"]
        ages = self.data_loader.data["AgeCategory"]
        conditions = []
        for stroke, age in zip(strokes, ages):
            if age >= 10 and stroke == 1:
                condition = 1
            else:
                condition = 0
            conditions.append(condition)
        self.data_loader.data["DangerousStroke"] = conditions
        print("Created DangerousStroke feature\n")

    def _age_bmi_interaction_feature(self):
        """
        Creates an age-BMI interaction feature based on the AgeCategory and BMI columns of the data.
        The age-BMI interaction feature is calculated by multiplying the values of the AgeCategory and BMI columns.
        The age-BMI interaction feature is stored in the "AgeBMI_Interaction" column of the data_loader object.
        """
        age = self.data_loader.data["AgeCategory"]
        bmi = self.data_loader.data["BMI"]
        condition = (age * bmi)
        self.data_loader.data["AgeBMI_Interaction"] = condition
        print("Created AgeBMI_Interaction feature\n")

    def _bmi_sleep_interaction_feature(self):
        """
        Creates a BMI-sleep interaction feature based on the BMI and SleepTime columns of the data.
        The BMI-sleep interaction feature is calculated by multiplying the values of the BMI and SleepTime columns.
        The BMI-sleep interaction feature is stored in the "BMISleep_Interaction" column of the data_loader object.
        """
        sleep = self.data_loader.data["SleepTime"]
        bmi = self.data_loader.data["BMI"]
        condition = (sleep * bmi)
        self.data_loader.data["BMISleep_Interaction"] = condition
        print("Created BMISleep_Interaction feature\n")

    def _age_gh_interaction_feature(self):
        """
        Creates an age-general health interaction feature based on the AgeCategory and GenHealth columns of the data.
        The age-general health interaction feature is calculated by multiplying the values of the AgeCategory and GenHealth columns.
        The age-general health interaction feature is stored in the "AgeHealth_Interaction" column of the data_loader object.
        """
        age = self.data_loader.data["AgeCategory"]
        general_health = self.data_loader.data["GenHealth"]
        condition = (age * general_health)
        self.data_loader.data["AgeHealth_Interaction"] = condition
        print("Created AgeHealth_Interaction feature\n")

    def _age_sleep_interaction_feature(self):
        """
        Creates an age-sleep interaction feature based on the AgeCategory and SleepTime columns of the data.
        The age-sleep interaction feature is calculated by multiplying the values of the AgeCategory and SleepTime columns.
        The age-sleep interaction feature is stored in the "AgeSleep_Interaction" column of the data_loader object.
        """
        age = self.data_loader.data["AgeCategory"]
        sleep = self.data_loader.data["SleepTime"]
        condition = (age * sleep)
        self.data_loader.data["AgeSleep_Interaction"] = condition
        print("Created AgeSleep_Interaction feature\n")

    def create_modified_features(self):

        self._bmi_class_feature()
        self._sleep_class_feature()

    def create_joined_features(self):

        self._bad_habits_score_feature()
        self._diseases_feature()
        self._poor_health_days_month()
        self._dangerous_age_stroke()

    def create_interaction_features(self):

        self._age_bmi_interaction_feature()
        self._bmi_sleep_interaction_feature()
        self._age_gh_interaction_feature()
        self._age_sleep_interaction_feature()


class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier.
    
    Parameters:
        - k (int): Number of neighbors to consider.
        - radius (float): Radius for radius search. Default is 30.
    
    Attributes:
        - k (int): Number of neighbors to consider.
        - radius (float): Radius for radius search.
        - nbrs (object): NearestNeighbors object.
        - y_train (array-like): Training labels.
        
    Methods:
        - fit(X_train, y_train): Fit the model to the training data.
        - score(X_val, y_val): Calculate the accuracy of the model on the validation data.
        - predict(X_val): Predict the labels for the validation data.
    """
    def __init__(self, k, radius=30):
        """
        Initialize the KNearestNeighbors object.
        
        Parameters:
            - k (int): Number of neighbors to consider.
            - radius (float): Radius for radius search. Default is 30.
        """
        self.k = k
        self.radius = radius
        self.nbrs = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.

        Parameters:
            - X_train (array-like): Training data.
            - y_train (array-like): Training labels.
        """
        # Initialize NearestNeighbors object with algorithm='ball_tree' for efficient nearest neighbor search
        self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(X_train)
        self.y_train = y_train

    def score(self, X_val, y_val):
        """
        Calculate the accuracy of the model on the validation data.

        Parameters:
            - X_val (array-like): Validation data.
            - y_val (array-like): Validation labels.

        Returns:
            - float: Accuracy of the model on the validation data.

        Raises:
            - ValueError: If the model has not been trained yet.
        """
        if self.nbrs is None:
            raise ValueError("Model has not been trained yet. Please call fit() before score().")

        # Perform radius search for each validation data point
        _, indices = self.nbrs.radius_neighbors(X_val, self.radius)

        correct_counts = 0
        total_counts = len(X_val)

        # Iterate through each validation data point and count correct predictions
        for i in range(total_counts):

            neighbor_labels = self.y_train[indices[i]]  # Get labels of neighbors within the radius

            while len(neighbor_labels) < self.k:  # If the number of neighbors is less than k
                self.radius += 3  # Increase the radius
                _, indices = self.nbrs.radius_neighbors(X_val, self.radius)  # Perform radius search again
                neighbor_labels = self.y_train[indices[i]] # Get labels of neighbors within the new radius

            predicted_label = np.bincount(neighbor_labels).argmax()  # Predict label based on majority vote
            if predicted_label == y_val[i]:  # Check if prediction matches the true label
                correct_counts += 1

        accuracy = correct_counts / total_counts  # Calculate accuracy
        return accuracy

    def predict(self, X_val):
        """
        Predict the labels for the validation data.

        Parameters:
            - X_val (array-like): Validation data.

        Returns:
            - array: Predicted labels for the validation data.

        Raises:
            - ValueError: If the model has not been trained yet.
        """
        if self.nbrs is None:
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")

        # Perform radius search for each validation data point
        _, indices = self.nbrs.radius_neighbors(X_val, self.radius)

        y_pred = []

        # Iterate through each validation data point and make predictions
        for i in range(len(X_val)):

            neighbor_labels = self.y_train[indices[i]]  # Get labels of neighbors within the radius

            while len(neighbor_labels) < self.k:  # If the number of neighbors is less than k
                self.radius += 3  # Increase the radius
                _, indices = self.nbrs.radius_neighbors(X_val, self.radius)  # Perform radius search again
                neighbor_labels = self.y_train[indices[i]]  # Get labels of neighbors within the new radius

            predicted_label = np.bincount(neighbor_labels).argmax()  # Predict label based on majority vote
            y_pred.append(predicted_label)

        return np.array(y_pred)


class ModelOptimization:
    """
    Class for optimizing the parameters of different classifiers.

    Parameters:
        - X_train (array-like): Training data.
        - y_train (array-like): Training labels.
        - X_val (array-like): Validation data.
        - y_val (array-like): Validation labels.

    Attributes:
        - X_train (array-like): Training data.
        - y_train (array-like): Training labels.
        - X_val (array-like): Validation data.
        - y_val (array-like): Validation labels.

    Methods:
        optimize_knn: Optimizes the parameters for K-Nearest Neighbors classifier.
        optimize_logistic_regression: Optimizes the parameters for Logistic Regression classifier.
        optimize_decision_tree: Optimizes the parameters for Decision Tree classifier.
        optimize_mlp: Optimizes the parameters for Multi-layer Perceptron (MLP) classifier.
    """

    def __init__(self, X_train, y_train, X_val, y_val):
        """
        Initialize the ModelOptimization object.

        Parameters:
            - X_train (array-like): Training data.
            - y_train (array-like): Training labels.
            - X_val (array-like): Validation data.
            - y_val (array-like): Validation labels.

        Attributes:
            - X_train (array-like): Training data.
            - y_train (array-like): Training labels.
            - X_val (array-like): Validation data.
            - y_val (array-like): Validation labels.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def optimize_knn(self, k_values):
        """
        Optimizes the parameters for K-Nearest Neighbors classifier.

        Parameters:
            k_values (list): List of k values to try.

        Returns:
            int: Best k value.
        """

        best_k = None
        best_accuracy = -1

        for k in k_values:

            knn = KNearestNeighbors(k)
            knn.fit(self.X_train, self.y_train)  # Fit the model to the training data
            accuracy = knn.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data
            print(f"k = {k}, Accuracy = {accuracy}")  # Print the accuracy for the current k value

            if accuracy > best_accuracy:  # Update the best k value if the accuracy is higher
                best_accuracy = accuracy
                best_k = k

        print("Best k value:", best_k)
        print("Best accuracy:", best_accuracy)
        return best_k

    def optimize_logistic_regression(self, C_values=(0.01, 0.1, 1.0, 10.0), penalty=('l1', 'l2')):
        """
        Optimizes the parameters for Logistic Regression classifier.

        Parameters:
            C_values (tuple): Values to try for regularization parameter C. Default is (0.01, 0.1, 1.0, 10.0).
            penalty (tuple): Penalty values to try. Default is ('l1', 'l2').

        Returns:
            tuple: Best parameters for Logistic Regression (C, penalty).
        """
        best_accuracy = -1
        best_c = None
        best_penalty = None

        for c in C_values:
            for penalty_selected in penalty:

                lr = LogisticRegression(C=c, penalty=penalty_selected, solver='saga', multi_class='auto', max_iter=1000)
                lr.fit(self.X_train, self.y_train)  # Fit the model to the training data
                accuracy = lr.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data

                print(f"C = {c}, Penalty = {penalty_selected}, Accuracy = {accuracy}")

                if accuracy > best_accuracy:  # Update the best parameters if the accuracy is higher
                    best_accuracy = accuracy
                    best_c = c
                    best_penalty = penalty_selected

        print("Best c value:", best_c)
        print("Best penalty:", best_penalty)
        print("Best accuracy:", best_accuracy)
        return best_c, best_penalty

    def optimize_decision_tree(self, max_depth_values=(None, 5, 10, 20)):
        """
        Optimizes the parameters for Decision Tree classifier.

        Parameters:
            max_depth_values (tuple): Values to try for max depth. Default is (None, 5, 10, 20).

        Returns:
            int: Best max depth value.
        """
        best_accuracy = -1
        best_max_depth = None

        for max_depth in max_depth_values:  # Iterate over the max depth values

            dt = DecisionTreeClassifier(max_depth=max_depth)
            dt.fit(self.X_train, self.y_train)  # Fit the model to the training data
            accuracy = dt.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data

            print(f"Max depth = {max_depth}, Accuracy = {accuracy}")

            if accuracy > best_accuracy:  # Update the best max depth value if the accuracy is higher
                best_accuracy = accuracy
                best_max_depth = max_depth

        print("Best max depth value:", best_max_depth)
        print("Best accuracy:", best_accuracy)
        return best_max_depth

    def optimize_mlp(self, population_size=20, max_generations=50, layer_range=(1, 100), activation=('logistic', 'tanh')):
        """
        Optimizes the parameters for Multi-layer Perceptron (MLP) classifier.

        Parameters:
            population_size (int): Number of individuals in the population. Default is 20.
            max_generations (int): Maximum number of generations. Default is 50.
            layer_range (tuple): Range of values to try for the number of neurons in the hidden layer. Default is (1, 100).
            activation (tuple): Activation functions to try. Default is ('logistic', 'tanh').

        Returns:
            tuple: Best parameters for MLP (neurons, activation).
        """
        # Initialize the population
        population = []
        # Generate random individuals
        for _ in range(population_size):

            neurons = random.randint(*layer_range)
            activation_selected = random.choice(activation)
            population.append((neurons, activation_selected))

        # Random evolutionary search
        for generation in range(max_generations):

            print(f"Generation {generation + 1}/{max_generations}")
            new_population = []  # Initialize the new population
            for i, (neurons, activation_selected) in enumerate(population):  # Iterate over the population

                # Skip the first iteration from generation 1 onwards since it is the best elements foun on the previous iteration
                if generation != 0 and i == 0:
                    new_population.append((best_params, best_accuracy))
                    continue

                print("Generation ", generation, " element ", i)  # Print the generation and element number
                mlp = MLPClassifier(hidden_layer_sizes=(neurons,), activation=activation_selected, early_stopping=True)
                mlp.fit(self.X_train, self.y_train)  # Fit the model to the training data
                accuracy = mlp.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data
                new_population.append(((neurons, activation_selected), accuracy))  # Add the individual to the new population
                print(f"Neurons: {neurons}, activation: {activation_selected}, Accuracy: {accuracy}")

            new_population.sort(key=lambda x: x[1], reverse=True)
            # Keep only the best element (concept of elitism in genetic algorithm)
            best_params, best_accuracy = new_population[0]
            population = [best_params]

            # Use the parameters of the best individual to bias the generation of new individuals
            best_neurons, best_activation = best_params

            # Break when only 2 individuals are left
            if population_size <= 2:
                break
            population_size -= 1

            # Generate the new population
            for _ in range(1, population_size):
                # Randomly generate neurons with a bias towards the best_neurons
                neurons = random.randint(np.ceil(best_neurons - 20) + 1, best_neurons + 20)
                # Randomly select activation function
                activation_selected = random.choice(activation)
                population.append((neurons, activation_selected))

        print("Best MLP parameters:", best_params)
        print("Best accuracy:", best_accuracy)
        return best_params


class CrossValidator:
    """
    Class for performing k-fold cross-validation on machine learning models.

    Parameters:
        k (int): Number of folds for cross-validation. Default is 5.

    Attributes:
        k (int): Number of folds for cross-validation.
        kf (object): KFold object for splitting the data.
        cm (object): ConfusionMatrix object for calculating confusion matrix.
        accuracy_scores (list): List of accuracy scores for each fold.
        sensitivity_scores (list): List of sensitivity scores for each fold.
        specificity_scores (list): List of specificity scores for each fold.

    Methods:
        cross_validate: Performs k-fold cross-validation on the model.
        evaluate_on_test_set: Evaluates the trained model on the test set.
    """

    def __init__(self, k=5):
        """
        Initialize the CrossValidator object.

        Parameters:
            k (int): Number of folds for cross-validation. Default is 5.

        Attributes:
            k (int): Number of folds for cross-validation.
            kf (object): KFold object for splitting the data.
            cm (object): ConfusionMatrix object for calculating confusion matrix.
            accuracy_scores (list): List of accuracy scores for each fold.
            sensitivity_scores (list): List of sensitivity scores for each fold.
            specificity_scores (list): List of specificity scores for each fold.
        """

        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True)
        self.cm = None
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []

    def cross_validate(self, model, X, y):
        """
        Performs k-fold cross-validation on the model.

        Parameters:
            model: Machine learning model.
            X (array-like): Features.
            y (array-like): Labels.

        Returns:
            tuple: Average accuracy, sensitivity, and specificity scores.
        """

        for train_index, val_index in self.kf.split(X):  # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)  # Fit the model to the training data
            y_pred = model.predict(X_val)  # Predict the labels for the validation data

            self.cm = ConfusionMatrix(actual_vector=list(y_val), predict_vector=list(y_pred))

            self.accuracy_scores.append(accuracy_score(y_val, y_pred))
            self.sensitivity_scores.append(float(self.cm.TPR_Macro))
            self.specificity_scores.append(float(self.cm.TNR_Macro))

        avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)

        return avg_accuracy, avg_sensitivity, avg_specificity

    def evaluate_on_test_set(self, model, X_test, y_test):
        """
        Evaluates the trained model on the test set.

        Parameters:
            model: Trained machine learning model.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            tuple: Accuracy, sensitivity, and specificity scores on the test set.
        """

        y_pred = model.predict(X_test)
        cm = ConfusionMatrix(actual_vector=list(y_test), predict_vector=list(y_pred))
        accuracy = cm.Overall_ACC
        sensitivity = cm.TPR_Macro
        specificity = cm.TNR_Macro
        return accuracy, sensitivity, specificity


class ModelBuilding:
    """
    Class for building, optimizing and evaluating machine learning models.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        X_val (array-like): Validation data.
        y_val (array-like): Validation labels.
        k (int): Number of folds for cross-validation. Default is 5.
        save_all (bool): Flag to save all models. Default is True.

    Attributes:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        X_val (array-like): Validation data.
        y_val (array-like): Validation labels.
        k (int): Number of folds for cross-validation.
        save_all (bool): Flag to save all models.
        best_model (object): Best performing model.
        best_model_name (str): Name of the best performing model.
        best_params (tuple): Best parameters for the best performing model.
        best_score (float): Best validation score.
        best_model_changed (bool): Flag to track if the best model changed.
        history (dict): Dictionary to store the validation scores of all models.

    Methods:
        build_models: Builds, optimizes and evaluates machine learning models.
        evaluate_best_model: Evaluates the best model on the test set.
        save_model: Saves the model to a file.
    """

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, k=5, save_all=True):
        """
        Initialize the ModelBuilding object.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Training labels.
            X_test (array-like): Test data.
            y_test (array-like): Test labels.
            X_val (array-like): Validation data.
            y_val (array-like): Validation labels.
            k (int): Number of folds for cross-validation. Default is 5.
            save_all (bool): Flag to save all models. Default is True.

        Attributes:
            X_train (array-like): Training data.
            y_train (array-like): Training labels.
            X_test (array-like): Test data.
            y_test (array-like): Test labels.
            X_val (array-like): Validation data.
            y_val (array-like): Validation labels.
            k (int): Number of folds for cross-validation.
            save_all (bool): Flag to save all models.
            best_model (object): Best performing model.
            best_model_name (str): Name of the best performing model.
            best_params (tuple): Best parameters for the best performing model.
            best_score (float): Best validation score.
            best_model_changed (bool): Flag to track if the best model changed.
            history (dict): Dictionary to store the validation scores of all models.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.k = k
        self.save_all = save_all
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = -1
        self.best_model_changed = False
        self.history = {}

    def build_models(self, model_name, models_dict, results_dict):
        """
        Builds, optimizes and evaluates machine learning models.

        Parameters:
            model_name (str): Name of the model to build.
            models_dict (dict): Dictionary containing the models to build.
            results_dict (dict): Dictionary to store the results of the models.

        Raises:
            ValueError: If the model type is not supported.
        """

        model_optimization = ModelOptimization(self.X_train, self.y_train, self.X_val, self.y_val)
        print("\nTraining", model_name, "model")
        for name, model_params in models_dict.items():  # Iterate over the models
            if model_name != name:
                continue
            model = model_params.pop('model')
            model_params_check = {}

            if model_name == "KNN":  # Optimize the parameters for K-Nearest Neighbors
                k = model_optimization.optimize_knn(model_params['k'])
                model_params_check['k'] = k
                params = {'k': k}
            elif model_name == "LogisticRegression":  # Optimize the parameters for Logistic Regression
                lr_params = model_optimization.optimize_logistic_regression(**model_params)
                model_params_check['C'] = lr_params[0]
                model_params_check['penalty'] = lr_params[1]
                params = lr_params
            elif model_name == "DecisionTree":  # Optimize the parameters for Decision Tree
                dt_params = model_optimization.optimize_decision_tree(**model_params)
                model_params_check['max_depth'] = dt_params
                params = dt_params
            elif model_name == "MLP":  # Optimize the parameters for Multi-layer Perceptron (MLP)
                mlp_params = model_optimization.optimize_mlp(**model_params)
                model_params_check['hidden_layer_sizes'] = (mlp_params[0],)
                model_params_check['activation'] = mlp_params[1]
                params = mlp_params
            else:
                raise ValueError("Model type is not supported.")

            model_instance = model(**model_params_check)  # Create an instance of the model with the optimized parameters
            model_instance.fit(self.X_train, self.y_train)  # Fit the model to the training data
            val_score = model_instance.score(self.X_val, self.y_val)  # Calculate the accuracy on the validation data
            self.history[str(name)] = val_score  # Store the validation score in the history dictionary

            if val_score > self.best_score:  # Update the best model if the validation score is higher
                print("\nNew best model found!")
                self.best_score = val_score
                self.best_model = model_instance
                self.best_model_name = name
                self.best_params = params
                self.best_model_checked = model
                self.best_model_params_checked = model_params_check
                self.best_model_changed = True  # Update flag when a new best model is found
            else:
                self.best_model_changed = False  # Reset flag if the best model didn't change

            if self.save_all:  # Save all models if the flag is set
                self.save_model(model_instance, name)

        self.kf_cv = CrossValidator(k=self.k)

        print(f"\nPreforming cross-validation on the {model_name} model:")

        # Performance of the model during cross-validation
        avg_accuracy_cv, avg_sensitivity_cv, avg_specificity_cv = self.kf_cv.cross_validate(model_instance,
                                                                                            self.X_train, self.y_train)

        print("Average accuracy during cross-validation:", avg_accuracy_cv)
        print("Average sensitivity during cross-validation:", avg_sensitivity_cv)
        print(f"Average specificity during cross-validation: {avg_specificity_cv}\n")

        print(f"\nPerformance of the {model_name} model on the Test set:")

        # Performance of the model on the test set
        accuracy_test, sensitivity_test, specificity_test = self.kf_cv.evaluate_on_test_set(model_instance, self.X_test,
                                                                                            self.y_test)
        print("Test set accuracy:", accuracy_test)
        print("Test set sensitivity:", sensitivity_test)
        print(f"Test set specificity: {specificity_test}\n")

        # Store the results in the results dictionary
        results_dict[model_name]['Accuracy'] = float(accuracy_test)
        results_dict[model_name]['Sensitivity'] = float(sensitivity_test)
        results_dict[model_name]['Specificity'] = float(specificity_test)

        # Print the best model and its parameters
        print("\nOptimization finished, history:\n")
        print("Model name\t\tAccuracy")
        for model, accuracy in self.history.items():
            print(f"{model}\t\t{accuracy}")
        print("\nBest performing model:", self.best_model_name)
        print("Best validation score:", self.best_score)
        print("Best parameters:", self.best_params)

        if not self.save_all:  # Save the best model if the flag is not set
            self.save_model(self.best_model, self.best_model_name)

    def save_model(self, model, filename):
        """
        Saves the model to a file.

        Parameters:
            model: Trained machine learning model.
            filename (str): Name of the file to save the model.
        """

        folder_path = "./models"
        full_path = os.path.join(folder_path, filename)
        print("Saving model as", filename)
        joblib.dump(model, full_path)


class BaggingClassifier:
    """
    A class for implementing the Bagging ensemble method with a base model.

    Parameters:
        base_model: Base machine learning model to use for bagging (Best Model).
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        n_straps (int): Number of bootstrap samples. Default is 100.
        k_fold (int): Number of folds for cross-validation. Default is 5.

    Attributes:
        base_model: Base machine learning model to use for bagging.
        n_straps (int): Number of bootstrap samples.
        k_fold (int): Number of folds for cross-validation.
        models (list): List of trained models.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        accuracy_scores (list): List of accuracy scores.
        sensitivity_scores (list): List of sensitivity scores.
        specificity_scores (list): List of specificity scores.
        avg_accuracy (float): Average accuracy score.
        avg_sensitivity (float): Average sensitivity score.
        avg_specificity (float): Average specificity score.

    Methods:
        examine_bagging: Fits the bagging ensemble on the training data with k fold cross-validation.
        predict: Predicts class labels for input data.
        evaluate: Evaluates the bagging ensemble on test data.
    """

    def __init__(self, base_model, X_train, y_train, X_test, y_test, n_straps=100, k_fold=5):

        self.base_model = base_model
        self.n_straps = n_straps
        self.k_fold = k_fold
        self.models = []
        self.X_train, self.y_train = X_train, y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.avg_accuracy = None
        self.avg_sensitivity = None
        self.avg_specificity = None

    def examine_bagging(self):
        """
        Fits the bagging ensemble on the training data with k fold cross-validation.

        Returns:
            tuple: Average accuracy, sensitivity, and specificity scores.
        """
        kfold = KFold(n_splits=self.k_fold, shuffle=True)
        for train_index, val_index in kfold.split(self.X_train):
            # Generate n_straps samples and train the models for the current fold
            self.models = []
            for _ in range(self.n_straps):
                # Create bootstrap sample with the available indices of the fold
                bootstrap_indices = np.random.choice(train_index, size=len(self.X_train[train_index]), replace=True)
                X_bootstrap = self.X_train[bootstrap_indices]
                y_bootstrap = self.y_train[bootstrap_indices]

                # Train base model on bootstrap sample
                model = clone(self.base_model)
                model.fit(X_bootstrap, y_bootstrap)
                self.models.append(model)

            y_pred = self.predict(self.X_train[val_index])

            self.cm = ConfusionMatrix(actual_vector=list(self.y_train[val_index]), predict_vector=list(y_pred))
            self.accuracy_scores.append(accuracy_score(self.y_train[val_index], y_pred))
            print(self.cm)
            self.sensitivity_scores.append(float(self.cm.TPR_Macro))
            self.specificity_scores.append(float(self.cm.TNR_Macro))

        self.avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        self.avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        self.avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)

        return self.avg_accuracy, self.avg_sensitivity, self.avg_specificity

    def predict(self, X):
        """
        Predicts class labels for input data.

        Parameters:
            X (array-like): Features.

        Returns:
            array-like: Predicted class labels.
        """
        # Aggregate predictions from all models
        predictions = np.zeros((len(X), self.n_straps))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        # Use majority voting to determine final prediction
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
        return final_predictions

    def evaluate(self, results_dict):
        """
        Evaluates the bagging ensemble on test data.

        Parameters:
            results_dict (dict): Dictionary to store the results of the models.
        """
        self.y_pred = self.predict(self.X_test)
        self.cm = ConfusionMatrix(actual_vector=list(self.y_test), predict_vector=list(self.y_pred))
        self.accuracy_scores = accuracy_score(self.y_test, self.y_pred)
        self.sensitivity_scores = float(self.cm.TPR_Macro)
        self.specificity_scores = float(self.cm.TNR_Macro)

        results_dict['Bagging']['Accuracy'] = float(self.accuracy_scores)
        results_dict['Bagging']['Sensitivity'] = float(self.sensitivity_scores)
        results_dict['Bagging']['Specificity'] = float(self.specificity_scores)

        print(f"Accuracy: {self.accuracy_scores}, Sensitivity: {self.sensitivity_scores}, Specificity: {self.specificity_scores}")


class AdaBoostClassifier:
    """
    A class for implementing the AdaBoost ensemble method with a base model.

    Parameters:
        best_model: Best model to use for AdaBoost.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        builder (object): Builder object with the configurations of the model.
        n_estimators (int): Number of weak learners. Default is 50.
        learning_rate (float): Learning rate shrinks the contribution of each weak learner. Default is 1.0.
        k_fold (int): Number of folds for cross-validation. Default is 5.

    Attributes:
        best_model: Best model to use for AdaBoost.
        n_estimators (int): Number of weak learners.
        learning_rate (float): Learning rate shrinks the contribution of each weak learner.
        k_fold (int): Number of folds for cross-validation.
        models (list): List of trained models.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        accuracy_scores (list): List of accuracy scores.
        sensitivity_scores (list): List of sensitivity scores.
        specificity_scores (list): List of specificity scores.
        builder (object): Builder object with the configurations of the model.
        avg_accuracy (float): Average accuracy score.

    Methods:
        train_adaboost: Trains the AdaBoost ensemble on the training data with k fold cross-validation.
        predict: Predicts class labels for input data.
        evaluate: Evaluates the AdaBoost ensemble on test data.
    """
    def __init__(self, best_model, X_train, y_train, X_test, y_test, builder, n_estimators=50, learning_rate=1.0, k_fold=5):

        self.best_model = best_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.k_fold = k_fold
        self.models = []
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.builder = builder
        self.avg_accuracy = None

    def train_adaboost(self):
        """
        Trains the AdaBoost ensemble on the training data with k fold cross-validation.

        Returns:
            float: Average accuracy score.
        """
        kfold = KFold(n_splits=self.k_fold, shuffle=True)
        for train_index, val_index in kfold.split(self.X_train):
            # Initialize weights
            w = np.ones(len(train_index)) / len(train_index)

            models = []
            for level in range(self.n_estimators):
                print("Level ", level, " of ", self.n_estimators-1)
                # Train base model with weighted samples

                base_model = None

                if (self.best_model == 'KNN'):

                    base_model = KNeighborsClassifier(n_neighbors=self.builder.best_model_params_checked['n_neighbors'])

                elif (self.best_model == 'LogisticRegression'):

                    base_model = LogisticRegression(C=self.builder.best_model_params_checked['C'],
                                                    penalty=self.builder.best_model_params_checked['penalty'])

                elif (self.best_model == 'DecisionTree'):

                    base_model = DecisionTreeClassifier(max_depth=self.builder.best_model_params_checked['max_depth'])

                elif (self.best_model == 'MLP'):

                    base_model = MLPClassifier(hidden_layer_sizes=self.builder.best_model_params_checked['hidden_layer_sizes'],
                                            activation=self.builder.best_model_params_checked['activation'], early_stopping=True)
                else:
                    raise ValueError("Best model not found.")

                # Train the model
                base_model.fit(self.X_train[train_index], self.y_train[train_index], sample_weight=w)

                print("Examining model")
                # Compute error
                y_pred = base_model.predict(self.X_train[train_index])
                err = np.sum(w * (y_pred != self.y_train[train_index])) / np.sum(w)

                print("Updating Adaboost model")
                # Compute alpha
                alpha = self.learning_rate * np.log((1 - err) / err)

                # Update weights
                w *= np.exp(alpha * (y_pred != self.y_train[train_index]))

                models.append((base_model, alpha))

            self.models.append(models)

            # Compute predictions for validation set
            y_pred = self.predict(val_index)

            # Evaluate performance
            self.accuracy_scores.append(accuracy_score(self.y_train[val_index], y_pred))
            print("Accuracy:", self.accuracy_scores[-1])

        self.avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)

        return self.avg_accuracy

    def predict(self, index):
        """
        Predicts class labels for input data.

        Parameters:
            index (array-like): Indices of the data to predict.

        Returns:
            array-like: Predicted class labels.
        """
        num_classes = len(np.unique(self.y_train))
        predictions = np.zeros((len(index), num_classes))
        for model, alpha in self.models[-1]:
            y_pred = model.predict(self.X_train[index])
            # Adjust the size of predictions array to match the expected size
            temp_predictions = np.zeros((len(index), num_classes))
            # Iterate over each sample and increment the corresponding class prediction
            for i, pred in enumerate(y_pred):
                temp_predictions[i, int(pred)] += alpha
            predictions += temp_predictions

        final_predictions = np.argmax(predictions, axis=1)
        return final_predictions

    def evaluate(self, results_dict):
        """
        Evaluates the AdaBoost ensemble on test data.

        Parameters:
            results_dict (dict): Dictionary to store the results of the models.

        Returns:
            float: Accuracy score.
        """
        num_classes = len(np.unique(self.y_train))
        predictions = np.zeros((len(self.y_test), num_classes))
        for model, alpha in self.models[-1]:
            y_pred = model.predict(self.X_test)
            temp_predictions = np.zeros((len(self.y_test), num_classes))
            for i, pred in enumerate(y_pred):
                temp_predictions[i, int(pred)] += alpha
            predictions += temp_predictions
        final_predictions = np.argmax(predictions, axis=1)

        # Calculate confusion matrix
        cm = ConfusionMatrix(actual_vector=list(self.y_test), predict_vector=list(final_predictions))
        sensitivity_boost = cm.TPR_Macro
        specificity_boost = cm.TNR_Macro

        # Calculated accuracy
        accuracy_boost = accuracy_score(self.y_test, final_predictions)
        print(f"\nTest Accuracy: {self.accuracy_scores}")

        # Calculated sensitivity
        print(f"Test Sensitivity: {sensitivity_boost}")

        # Calculated specificity
        print(f"Test Specificity: {specificity_boost}\n")

        # Store the results in the results dictionary
        results_dict['AdaBoost']['Accuracy'] = float(accuracy_boost)
        results_dict['AdaBoost']['Sensitivity'] = float(sensitivity_boost)
        results_dict['AdaBoost']['Specificity'] = float(specificity_boost)


class CNN:
    """
    A class to implement a Convolutional Neural Network (CNN) for binary classification.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        X_val (array-like): Validation data.
        y_val (array-like): Validation labels.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes.
        epochs (int): Number of epochs. Default is 10.
        batch_size (int): Batch size. Default is 32.

    Attributes:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        X_val (array-like): Validation data.
        y_val (array-like): Validation labels.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        model (object): CNN model.

    Methods:
        build_model: Builds the CNN model.
        train: Trains the CNN model.
        evaluate: Evaluates the CNN model.
    """

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, input_shape, num_classes, epochs=10, batch_size=32):
        """
        Initialize the CNN object.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Training labels.
            X_test (array-like): Test data.
            y_test (array-like): Test labels.
            X_val (array-like): Validation data.
            y_val (array-like): Validation labels.
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of classes.
            epochs (int): Number of epochs. Default is 10.
            batch_size (int): Batch size. Default is 32.

        Attributes:
            X_train (array-like): Training data.
            y_train (array-like): Training labels.
            X_test (array-like): Test data.
            y_test (array-like): Test labels.
            X_val (array-like): Validation data.
            y_val (array-like): Validation labels.
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of classes.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            model (object): CNN model.
        """
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.X_val, self.y_val = X_val, y_val
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the CNN model.

        Returns:
            object: CNN model.
        """

        # Define the CNN model
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(self.input_shape[0], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self):
        """
        Trains the CNN model.

        Returns:
            object: Training history.
        """

        # Ensure target labels are one-dimensional
        y_train = self.y_train.squeeze()
        y_val = self.y_val.squeeze()

        # Fit the model to the training data and validate on the validation data
        history = self.model.fit(self.X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 verbose=1, validation_data=(self.X_val, y_val))
        return history

    def evaluate(self, results_dict):
        """
        Evaluates the CNN model.

        Parameters:
            results_dict (dict): Dictionary to store the results of the models.
        """

        # Ensure target labels are one-dimensional
        y_test = self.y_test.squeeze()

        # Get raw predictions from the model
        y_pred_prob = self.model.predict(self.X_test)

        # Convert probabilities to class labels using a threshold (e.g., 0.5)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])

        # Calculate specificity
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Store the results in the results dictionary
        results_dict['CNN']['Accuracy'] = float(accuracy)
        results_dict['CNN']['Sensitivity'] = float(sensitivity)
        results_dict['CNN']['Specificity'] = float(specificity)

        print("\nCNN Accuracy:", accuracy)
        print("CNN Sensitivity:", sensitivity)
        print("CNN Specificity:", specificity)


class ClusteringModel:
    """
    A class to perform various clustering algorithms and visualize their results.

    Parameters:
        data_train (DataFrame): The training dataset.
        data_test (DataFrame): The testing dataset.
        n_clusters (int): The number of clusters to form.

    Attributes:
        data_train (DataFrame): The training dataset.
        data_test (DataFrame): The testing dataset.
        n_clusters (int): The number of clusters to form.
        train_labels (array): Labels generated by clustering algorithms for training data.
        test_labels (array): Labels generated by clustering algorithms for testing data.

    Methods:
        hierarchical_clustering: Performs hierarchical clustering and visualizes the dendrogram.
        k_means: Performs K-Means clustering and visualizes the clusters.
        gaussian_mixture_model: Performs Gaussian Mixture Model clustering and visualizes the clusters.
        optics: Performs OPTICS clustering and visualizes the clusters.
        plot_clusters: Plots clusters in 2D using Principal Component Analysis.
        perform_clustering: Performs all clustering methods and visualizes their results.
    """

    def __init__(self, data_train, data_test, n_clusters):
        """
        Initialize the ClusteringModel object.

        Parameters:
            data_train (DataFrame): The training dataset.
            data_test (DataFrame): The testing dataset.
            n_clusters (int): The number of clusters to form.

        Attributes:
            data_train (DataFrame): The training dataset.
            data_test (DataFrame): The testing dataset.
            n_clusters (int): The number of clusters to form.
            train_labels (array): Labels generated by clustering algorithms for training data.
            test_labels (array): Labels generated by clustering algorithms for testing data.
        """

        self.data_train = data_train
        self.data_test = data_test
        self.n_clusters = n_clusters
        self.train_labels = None
        self.test_labels = None

    def hierarchical_clustering(self):
        """
        Performs hierarchical clustering and visualizes the dendrogram.
        This method performs hierarchical clustering on a random sample of the training data and visualizes the dendrogram.
        """

        print("\nHierarchical Clustering:")
        # Only use 50% of the data for visualization because of the large size
        random_sample = self.data_test.sample(n=int(0.5 * len(self.data_test)), replace=False)
        # Plot dendrogram
        sch.dendrogram(sch.linkage(random_sample, method='ward'), color_threshold=30)
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

    def k_means(self):
        """
        Performs K-Means clustering and visualizes the clusters.
        This method performs K-Means clustering on the testing data and visualizes the clusters in 2D.
        """

        print("\nK-Means Clustering:")
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit_predict(self.data_train)
        self.test_labels = kmeans.predict(self.data_test)
        self.plot_clusters(self.data_test, self.test_labels)

    def gaussian_mixture_model(self):
        """
        Performs Gaussian Mixture Model clustering and visualizes the clusters.
        This method performs Gaussian Mixture Model clustering on the testing data and visualizes the clusters in 2D.
        """

        print("\nGaussian Mixture Model:")
        # Only use 50% of the data for visualization because of the large size
        random_sample = self.data_train.sample(n=int(0.5 * len(self.data_train)), replace=False)
        gmm = GaussianMixture(n_components=self.n_clusters)
        gmm.fit_predict(random_sample)
        self.test_labels = gmm.predict(self.data_test)
        self.plot_clusters(self.data_test, self.test_labels)

    def optics(self):
        """
        Performs OPTICS clustering and visualizes the clusters.
        This method performs OPTICS clustering on the testing data and visualizes the clusters in 2D.
        """

        print("\nOPTICS Clustering:")
        # Only use 50% of the data for visualization because of the large size
        random_sample = self.data_test.sample(n=int(0.5 * len(self.data_test)), replace=False)
        optics = OPTICS(min_samples=3)
        self.test_labels = optics.fit(random_sample)
        self.plot_clusters(random_sample, self.test_labels.labels_)

    def plot_clusters(self, data, labels):
        """
        Plots clusters in 2D using Principal Component Analysis.

        Parameters:
            data (array): The data to plot.
            labels (array): The labels of the clusters.
        """

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        # Plot clusters in 2D
        sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette='viridis')
        plt.title("Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

    def perform_clustering(self):
        """
        Performs all clustering methods and visualizes their results.
        """

        self.hierarchical_clustering()
        self.k_means()
        self.gaussian_mixture_model()
        self.optics()


class BestFeatureSelector:
    """
    A class to perform feature selection using Sequential Backward Feature Selection.

    Parameters:
        X_train (DataFrame): Training data.
        y_train (array): Training labels.
        X_test (DataFrame): Test data.
        y_test (array): Test labels.
        X_val (DataFrame): Validation data.
        y_val (array): Validation labels.
        base_model: Base machine learning model to use for feature selection.

    Attributes:
        X_train (DataFrame): Training data.
        y_train (array): Training labels.
        X_val (DataFrame): Validation data.
        y_val (array): Validation labels.
        X_test (DataFrame): Test data.
        y_test (array): Test labels.
        base_model: Base machine learning model to use for feature selection.
        selected_features_idx (array): Indices of the selected features.
    """

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, base_model):
        """
        Initialize the BestFeatureSelector object.

        Parameters:
            X_train (DataFrame): Training data.
            y_train (array): Training labels.
            X_test (DataFrame): Test data.
            y_test (array): Test labels.
            X_val (DataFrame): Validation data.
            y_val (array): Validation labels.
            base_model: Base machine learning model to use for feature selection.

        Attributes:
            X_train (DataFrame): Training data.
            y_train (array): Training labels.
            X_val (DataFrame): Validation data.
            y_val (array): Validation labels.
            X_test (DataFrame): Test data.
            y_test (array): Test labels.
            base_model: Base machine learning model to use for feature selection.
            selected_features_idx (array): Indices of the selected features.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.base_model = base_model
        self.selected_features_idx = None

    def select_features(self):
        """
        Perform sequential backward feature selection.

        This method selects the best set of features using backward feature selection.
        """
        # Define the Sequential Feature Selector
        sfs = SequentialFeatureSelector(self.base_model, k_features='best', forward=False, floating=False,
                                        scoring='accuracy', verbose=True, n_jobs=8, cv=2)

        # Perform Sequential Backward Feature Selection
        sfs = sfs.fit(self.X_train, self.y_train)

        # Get the selected feature indices
        self.selected_features_idx = sfs.k_feature_idx_

    def evaluate(self, results_dict):
        """
        Evaluate the best model with the selected features.
        This method evaluates the best model using the selected features on the test set.

        Parameters:
            results_dict (dict): Dictionary to store the results of the models.
        """

        X_train_selected = self.X_train.iloc[:, list(self.selected_features_idx)]
        X_train_selected = X_train_selected.values

        X_test_selected = self.X_test.iloc[:, list(self.selected_features_idx)]
        X_test_selected = X_test_selected.values

        # Train the model with the selected features
        clf = self.base_model.fit(X_train_selected, self.y_train)

        # Predictions on test set
        y_test_pred = clf.predict(X_test_selected)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_test_pred)

        # Calculate sensitivity
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])

        # Calculate specificity
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        # Store the results in the results dictionary
        results_dict['FeatureSelection']['Accuracy'] = float(accuracy)
        results_dict['FeatureSelection']['Sensitivity'] = float(sensitivity)
        results_dict['FeatureSelection']['Specificity'] = float(specificity)

        print(f"\nFeature Selection Accuracy: {accuracy}")
        print(f"Feature Selection Sensitivity: {sensitivity}")
        print(f"Feature Selection Specificity: {specificity}")


class DataDemonstration:
    """
    A class to demonstrate the results of the models.

    Parameters:
        results_dict (dict): Dictionary containing the results of the models.

    Attributes:
        data_results (dict): Dictionary containing the results of the models.
        data_specifications (dict): Dictionary containing the specifications of the models.
        df_results (DataFrame): DataFrame containing the results of the models.
        df_specifications (DataFrame): DataFrame containing the specifications of the models.
    """

    def __init__(self, results_dict):
        """
        Initialize the DataDemonstration object.

        Parameters:
            results_dict (dict): Dictionary containing the results of the models.

        Attributes:
            data_results (dict): Dictionary containing the results of the models.
            data_specifications (dict): Dictionary containing the specifications of the models.
            df_results (DataFrame): DataFrame containing the results of the models.
            df_specifications (DataFrame): DataFrame containing the specifications of the models.
        """

        self.data_results = {
            'Model': ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F', 'Model G', 'Model H'],
            'Category': ['All Models', 'Supervised Learning', 'Supervised Learning', 'Supervised Learning', 'Bagging',
                         'Boosting', 'Deep Learning', 'Feature Selection'],
            'Accuracy': [results_dict['KNN']['Accuracy'], results_dict['LogisticRegression']['Accuracy'],
                         results_dict['DecisionTree']['Accuracy'], results_dict['MLP']['Accuracy'],
                         results_dict['Bagging']['Accuracy'], results_dict['AdaBoost']['Accuracy'],
                         results_dict['CNN']['Accuracy'], results_dict['FeatureSelection']['Accuracy']],
            'Sensitivity': [results_dict['KNN']['Sensitivity'], results_dict['LogisticRegression']['Sensitivity'],
                            results_dict['DecisionTree']['Sensitivity'], results_dict['MLP']['Sensitivity'],
                            results_dict['Bagging']['Sensitivity'], results_dict['AdaBoost']['Sensitivity'],
                            results_dict['CNN']['Sensitivity'], results_dict['FeatureSelection']['Sensitivity']],
            'Specificity': [results_dict['KNN']['Specificity'], results_dict['LogisticRegression']['Specificity'],
                            results_dict['DecisionTree']['Specificity'], results_dict['MLP']['Specificity'],
                            results_dict['Bagging']['Specificity'], results_dict['AdaBoost']['Specificity'],
                            results_dict['CNN']['Specificity'], results_dict['FeatureSelection']['Specificity']],
            'Validation': ['2FCV', '2FCV', '2FCV', '2FCV', '2FCV', '2FCV', '2FCV', '2FCV'],
            'Remarks': ['Worst Performance Overall', 'Bad Accuracy but decent Sensibility and Specificity',
                        'Best Accuracy of the Supervised Models', 'Takes the most time to execute and has bad performance',
                        'Improved the best supervised model the most', 'Improved the best model a little',
                        'Best Sensitivity with decent Accuracy and Specificity', 'Best Specificity with decent Accuracy']
        }

        self.data_specifications = {
            'Model': ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F', 'Model G', 'Model H'],
            'Type': ['KNN', 'Logistic Regression', 'Decision Tree', 'MLP', 'Bagging of the Decision Tree',
                     'AdaBoost of the Decision Tree', 'CNN', 'Decision Tree with Feature Selection'],
            'Parameters': ['Number Neighbours: 3', 'C: 1.0, Penalty: l2', 'Max depth: None',
                           'Neurons: 218, Activation: logistic', 'Model: Decision Tree', 'Model: Decision Tree',
                           '8 layers without residual connection', 'Model: Decision Tree'],
        }

        # Convert data to DataFrames
        self.df_results = pd.DataFrame(self.data_results)
        self.df_specifications = pd.DataFrame(self.data_specifications)

    def tables(self):
        """
        Print the results and specifications tables.
        """

        # Format columns for results table
        self.df_results['Accuracy'] = self.df_results['Accuracy'].map('{:.2f}'.format)
        self.df_results['Sensitivity'] = self.df_results['Sensitivity'].map('{:.2f}'.format)
        self.df_results['Specificity'] = self.df_results['Specificity'].map('{:.2f}'.format)

        # Create results table
        results_table = pd.DataFrame({
            'Model': self.df_results['Model'],
            'Category': self.df_results['Category'],
            'Accuracy': self.df_results['Accuracy'],
            'Sensitivity': self.df_results['Sensitivity'],
            'Specificity': self.df_results['Specificity'],
            'Remarks': self.df_results['Remarks']
        })

        # Create specifications table
        specifications_table = pd.DataFrame({
            'Model': self.df_specifications['Model'],
            'Type': self.df_specifications['Type'],
            'Parameters': self.df_specifications['Parameters']
        })

        # Set display options
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        # Print results table
        print("\nResults Table:\n")
        print(results_table)
        print()

        # Print specifications table
        print("\nSpecifications Table:\n")
        print(specifications_table)

        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')


# %% 1- Pre Processing and EDA

path = 'data/heart_2020.csv'

# Initialize DataManipulator class
data_loader = DataManipulator(path, 'HeartDisease')

# Initialize DataPreProcessing class with the dataset
# Process all the data to numeric values and determine the range of values for each variable
data_preprocessing = DataPreProcessing(data_loader)

# Initialize DataVisualization class with the dataset
data_visualization = DataVisualization(data_loader, ['correlation', 'box', 'barh'])

# Visualization of the outliers (box plots) and all the histograms
# data_visualization.plot_all_features()
# data_visualization.plots(['box'])

# Initialize DataCleaning class with the dataset
data_cleaner = DataCleaning(data_loader)

# Verify the presence of missing values, duplicated data and outliers and clean the data
data_cleaner.handle_missing_values()
data_cleaner.remove_duplicates()
data_cleaner.detect_and_remove_outliers()

print("\nCleansed Dataset:")
print(data_loader.data.info)

# Save the cleaned dataset to a new csv file
data_loader.data.to_csv('data/heart_2020_cleaned.csv', index=False)

# Visualization of all the histograms, correlation and barh plots
# data_visualization.plot_all_features()
# data_visualization.plots(['correlation', 'barh'])

# Initialize DimensionalityReduction class with the dataset
dr = DimensionalityReduction(data_loader)

# Compute and plot PCA projection
# dr.plot_projection(dr.compute_pca(), 'PCA Projection')
# Compute and plot UMAP projection
# dr.plot_projection(dr.compute_umap(), 'UMAP Projection')

# %% 2- Hypothesis Testing

# Initialize the HypothesisTester class with the data
tester = HypothesisTester(data_loader)

# Perform normality analysis, first by normality test and then by visual checking using a Q-Q plot
# tester.test_normality()
# tester.qq_plots()

# After the analysis of the normality test and Q-Q plots we decided the distribution of variables
# We found from the Q-Q plots that the only normal distributed variables are BMI_with_HD and BMI_without_HD
# tester.distribute_normality_data()

# Perform the hypothesis tests
# tester.perform_tests()

# %% 3- Feature Creation

# Initialize the FeatureCreation class with the data
feature_creator = FeatureCreation(data_loader)

print("\nFeatures Created:\n")

# Create the features
feature_creator.create_modified_features()
feature_creator.create_joined_features()
feature_creator.create_interaction_features()

# Save the new dataset to a new csv file
data_loader.data.to_csv('data/heart_2020_final.csv', index=False)

# Visualization of all the final histograms and correlation plot
# data_visualization.plot_all_features()
# data_visualization.plots(['correlation'])

# %% 4- Model Building

# Initialize the SMOTEENN class for oversampling and undersampling
smote_enn = SMOTEENN(random_state=42)
undersample = RandomUnderSampler(random_state=42)

# Divide the data into features and target variable
X = data_loader.data.drop(columns=[data_loader.target])
y = data_loader.data[data_loader.target]

# Split the data into training and testing sets
X_train_under, X_test, y_train_under, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print class distribution before resampling
print("Class distribution before resampling:")
print(y_train_under.value_counts())

# Resample the training data
X_train, y_train = undersample.fit_resample(X_train_under, y_train_under)

print("\nClass distribution after resampling:")
print(y_train.value_counts())

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model dictionary for the optimization
models_dict = {
    "KNN": {"model": KNearestNeighbors, "k": (3, 5, 7)},
    "LogisticRegression": {"model": LogisticRegression, "C_values": (0.01, 0.1, 1.0, 10.0), "penalty": (None, 'l2')},
    "DecisionTree": {"model": DecisionTreeClassifier, "max_depth_values": (None, 5, 10, 20)},
    "MLP": {"model": MLPClassifier, "population_size": 5, "max_generations": 20, "layer_range": (150, 300),
            "activation": ("tanh", "logistic", "relu")}
}

# Define the results dictionary
results_dict = {
    "KNN": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "LogisticRegression": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "DecisionTree": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "MLP": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "Bagging": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "AdaBoost": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "CNN": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0},
    "FeatureSelection": {"Accuracy": 0.0, "Sensitivity": 0.0, "Specificity": 0.0}
}

# Initialize the ModelBuilding class with the training, testing and validation data
builder = ModelBuilding(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
                        np.array(X_val), np.array(y_val))

# KNN Algorithm
builder.build_models("KNN", models_dict, results_dict)

# Supervised Learning Algorithms
builder.build_models("LogisticRegression", models_dict, results_dict)
builder.build_models("DecisionTree", models_dict, results_dict)
builder.build_models("MLP", models_dict, results_dict)

# Serialize the builder object
with open('builder.pkl', 'wb') as f:
    pickle.dump(builder, f)

# Deserialize the builder object
with open('builder.pkl', 'rb') as f:
    builder = pickle.load(f)

# Name of the best model
best_model_name = None

# Check the best model
if builder.best_model_checked == KNearestNeighbors:
    best_model_name = 'KNN'
elif builder.best_model_checked == LogisticRegression:
    best_model_name = 'LogisticRegression'
elif builder.best_model_checked == DecisionTreeClassifier:
    best_model_name = 'DecisionTree'
elif builder.best_model_checked == MLPClassifier:
    best_model_name = 'MLP'

print("\nBest model:", best_model_name)
print("Best model parameters:", builder.best_model_params_checked)

# Ensemble Learning Models

# Initialize the BaggingClassifier class with the best model
bagging = BaggingClassifier(builder.best_model_checked(**builder.best_model_params_checked), np.array(X_train),
                            np.array(y_train), np.array(X_test), np.array(y_test))

# Examine the bagging ensemble
bagging.examine_bagging()
# Evaluate the bagging ensemble on the test set
bagging.evaluate(results_dict)

# Initialize the AdaBoostClassifier class with the best model
adaboost = AdaBoostClassifier(best_model_name, np.array(X_train), np.array(y_train), np.array(X_test),
                              np.array(y_test), builder)
# Train the AdaBoost ensemble
adaboost.train_adaboost()
# Evaluate the AdaBoost ensemble on the test set
adaboost.evaluate(results_dict)

# Deep Learning Model

# Define the input shape and number of classes
input_shape = (X_train.shape[1], 1)
num_classes = len(np.unique(y_train))

# Reshape the data for CNN
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

# Create an instance of CNN
cnn = CNN(X_train_cnn, y_train, X_test_cnn, y_test, X_val, y_val, input_shape, num_classes)

# Print the summary of the CNN model
cnn.model.summary()

# Train the CNN model
history = cnn.train()

# Evaluate the CNN model
cnn.evaluate(results_dict)

# Clustering Model

# Initialize the ClusteringModel class with the training and testing data
clustering_model = ClusteringModel(X_train, X_test, 10)
# Perform clustering and visualize the results
clustering_model.perform_clustering()

# Feature Selection

# Initialize the BestFeatureSelector class with the training, testing and validation data
feature_selector = BestFeatureSelector(X_train, y_train, X_test, y_test, X_val, y_val,
                                       builder.best_model_checked(**builder.best_model_params_checked))
# Select the best features
feature_selector.select_features()

# Serialize the feature_selector object
with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(feature_selector, f)

print("\nThe selected features were: ", feature_selector.selected_features_idx)

# Evaluate the model with the selected features
feature_selector.evaluate(results_dict)

# Data Demonstration

# Initialize the DataDemonstration class with the results dictionary
demonstration = DataDemonstration(results_dict)
# Print the results and specifications tables
demonstration.tables()
