# %% 0- Classes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.callbacks import EarlyStopping
from keras.src.layers import GlobalAveragePooling2D
from plotly.figure_factory._dendrogram import sch
from sklearn.cluster import KMeans, OPTICS
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import umap
from scipy.stats import ttest_ind, probplot, shapiro
import statsmodels.stats.api as sms
from pycm import ConfusionMatrix
import joblib
import os
import pickle
from tensorflow.keras.layers import Dense, ZeroPadding2D


# import warnings
# warnings.filterwarnings("ignore")


class DataLoader:
    """
    Generic Class responsible for loading the dataset

    Attributes:
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

        Args:
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

        Args:
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

    Args:
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

        Args:
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


class DataPreProcessing:
    """
    Class for performing data preprocessing tasks, mostly encoding.

    Args:
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

        Args:
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

        Args:
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

    Args:
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

        Args:
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
    
    Args:
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
            perm_sorted_idx = result.importances_mean.argsort()

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

    Args:
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

        Args:
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

        Args:
        - n_components: The number of components to keep.

        Returns:
        - The projected data after PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Perform Uniform Manifold Approximation and Projection (UMAP) on the data.

        Args:
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

    Args:
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

        Args:
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


class ModelOptimization:

    def __init__(self, X_train, y_train, X_val, y_val):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def optimize_knn(self, k_values):

        best_k = None
        best_accuracy = -1

        for k in k_values:

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            accuracy = knn.score(self.X_val, self.y_val)
            print(f"k = {k}, Accuracy = {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        print("Best k value:", best_k)
        print("Best accuracy:", best_accuracy)
        return best_k

    def optimize_logistic_regression(self, C_values=(0.1, 1.0, 10.0), penalty=('l1', 'l2')):
        """
        Optimizes the parameters for Logistic Regression classifier.

        Parameters:
            C_values (tuple): Values to try for regularization parameter C. Default is (0.1, 1.0, 10.0).
            penalty (tuple): Regularization penalty to try. Default is ('l1', 'l2').

        Returns:
            tuple: Best parameters for Logistic Regression (C, penalty).
        """
        best_accuracy = -1
        best_c = None
        best_penalty = None
        for c in C_values:
            for penalty_selected in penalty:
                lr = LogisticRegression(C=c, penalty=penalty_selected, solver='saga', multi_class='auto', max_iter=1000)
                lr.fit(self.X_train, self.y_train)
                accuracy = lr.score(self.X_val, self.y_val)
                print(f"C = {c}, Penalty = {penalty_selected}, Accuracy = {accuracy}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_c = c
                    best_penalty = penalty_selected
        print("Best c value:", best_c)
        print("Best penalty:", best_penalty)
        print("Best accuracy:", best_accuracy)
        return best_c, best_penalty

    def optimize_decision_tree(self, max_depth_values=(None, 10, 20)):
        """
        Optimizes the parameters for Decision Tree classifier.

        Parameters:
            max_depth_values (tuple): Values to try for maximum depth. Default is (None, 10, 20).

        Returns:
            tuple: Best parameters for Decision Tree (max depth).
        """
        best_accuracy = -1
        best_max_depth = None
        for max_depth in max_depth_values:
            dt = DecisionTreeClassifier(max_depth=max_depth)
            dt.fit(self.X_train, self.y_train)
            accuracy = dt.score(self.X_val, self.y_val)
            print(f"Max depth = {max_depth}, Accuracy = {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_max_depth = max_depth
        print("Best max depth value:", best_max_depth)
        print("Best accuracy:", best_accuracy)
        return best_max_depth


class CrossValidator:

    def __init__(self, k=5):
        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True)
        self.cm = None
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []

    def cross_validate(self, model, X, y):
        for train_index, val_index in self.kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

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

    def __init__(self, X_train, y_train, X_test, y_test, k=5, validation_size=0.2, save_all=True):

        self.X_train_complete = X_train
        self.y_train_complete = y_train
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train,
                                                                              test_size=validation_size)
        self.X_test = X_test
        self.y_test = y_test
        self.k = k
        self.save_all = save_all
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = -1
        self.history = {}

    def build_models(self, models_dict):

        model_optimization = ModelOptimization(self.X_train, self.y_train, self.X_val, self.y_val)

        for name, model_params in models_dict.items():

            print("\nTraining", name, "model")
            model = model_params.pop('model')
            model_params_check = {}

            if model == KNeighborsClassifier:

                k = model_optimization.optimize_knn(model_params['n_neighbors'])
                model_params_check['n_neighbors'] = k
                params = {'n_neighbors': k}

            elif model == LogisticRegression:

                lr_params = model_optimization.optimize_logistic_regression(**model_params)
                model_params_check['C'] = lr_params[0]
                model_params_check['penalty'] = lr_params[1]
                params = lr_params

            elif model == DecisionTreeClassifier:

                dt_params = model_optimization.optimize_decision_tree(**model_params)
                model_params_check['max_depth'] = dt_params
                params = dt_params

            else:
                raise ValueError("Model type is not supported.")

            # Check the performance of the optimized model parameters on the validation data
            model_instance = model(**model_params_check)
            model_instance.fit(self.X_train, self.y_train)
            val_score = model_instance.score(self.X_val, self.y_val)
            self.history[str(name)] = val_score

            if val_score > self.best_score:
                print("New best model found!")
                self.best_score = val_score
                self.best_model = model_instance
                self.best_model_name = name
                self.best_params = params
                self.best_model_checked = model
                self.best_model_params_checked = model_params_check

            if self.save_all:
                self.save_model(model_instance, name)

        print("Optimization finished, history:\n")
        print("Model name\t\tAccuracy")  # Print header
        for model, accuracy in self.history.items():  # Print table rows
            print(f"{model}\t\t{accuracy}")
        print("\nBest performing model:", self.best_model_name)
        print("Best validation score:", self.best_score)
        print("Best parameters:", self.best_params)

        if not self.save_all:
            self.save_model(self.best_model, self.best_model_name)

        # Evaluate best found model using CrossValidator
        self.kf_cv = CrossValidator(k=self.k)
        self.avg_accuracy, self.avg_sensitivity, self.avg_specificity = self.kf_cv.cross_validate(
            self.best_model_checked(**self.best_model_params_checked), self.X_train_complete, self.y_train_complete)

        print("Average accuracy during cross-validation:", self.avg_accuracy)
        print("Average sensitivity during cross-validation:", self.avg_sensitivity)
        print("Average specificity during cross-validation:", self.avg_specificity)

        return self.history, self.avg_accuracy, self.avg_sensitivity, self.avg_specificity

    def evaluate_best_model(self):

        print("\nEvaluating best model on test set!")
        test_score = self.best_model.score(self.X_test, self.y_test)
        print("Test set score:", test_score)

    def save_model(self, model, filename):

        folder_path = "./models"
        full_path = os.path.join(folder_path, filename)
        print("Saving model as", filename)
        joblib.dump(model, full_path)


# def KNearestNeighbors(X_val, X_train, y_val, y_train, k):
#
#     correct_count = 0
#
#     for i in range(len(X_val)):
#         print(f"Processing instance {i} out of {len(X_val)}")
#
#         distances = [np.sqrt(np.sum((X_val[i] - x_train) ** 2)) for x_train in X_train]
#         nearest_indices = np.argsort(distances)[:k]
#         nearest_labels = [y_train[index] for index in nearest_indices]
#         predicted_label = max(set(nearest_labels), key=nearest_labels.count)
#
#         if predicted_label == y_val[i]:
#             correct_count += 1
#
#     accuracy = correct_count / len(X_val)
#     return accuracy


class BaggingClassifier:
    """
    Performs bagging with a provided model.
    """

    def __init__(self, base_model, X_train, y_train, X_test, y_test, n_straps=100, k_fold=5):
        """
        Initializes the BaggingClassifier object.

        Parameters:
            base_model: Base machine learning model to use for bagging.
            n_straps (int): Number of bootstrap samples. Default is 10.
            k_fold (int): Number of folds for cross-validation. Default is 5.
        """
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

    def examine_bagging(self):
        """
        Fits the bagging ensemble on the training data with k fold cross-validation.
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
            X (array-like): Input features.

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

    def evaluate(self):
        """
        Evaluates the bagging ensemble on test data.

        Returns:
            tuple: Average accuracy, sensitivity, and specificity scores.
        """
        self.y_pred = self.predict(self.X_test)
        self.cm = ConfusionMatrix(actual_vector=list(self.y_test), predict_vector=list(self.y_pred))
        self.accuracy_scores = accuracy_score(self.y_test, self.y_pred)
        self.sensitivity_scores = float(self.cm.TPR_Macro)
        self.specificity_scores = float(self.cm.TNR_Macro)
        print(
            f"Accuracy: {self.accuracy_scores}, Sensitivity: {self.sensitivity_scores}, Specificity: {self.specificity_scores}")
        return self.accuracy_scores, self.sensitivity_scores, self.specificity_scores


class TransferLearningModel:
    """
    A class for transfer learning with pre-trained models.

    Attributes:
        train_data (DataFrame): The training data.
        train_labels (array-like): The training labels.
        test_data (DataFrame): The test data.
        test_labels (array-like): The test labels.
        num_classes (int): The number of classes in the classification task.

    Methods:
        preprocess_data(): Preprocesses the input data for transfer learning.
        pad_images(images): Pads input images to meet the minimum size requirement.
        transfer_learning(): Performs transfer learning using MobileNetV2.
    """

    def __init__(self, train_data, train_labels, test_data, test_labels, num_classes):
        """
        Initializes the TransferLearningModel with input data and parameters.

        Args:
            train_data (DataFrame): The training data.
            train_labels (array-like): The training labels.
            test_data (DataFrame): The test data.
            test_labels (array-like): The test labels.
            num_classes (int): The number of classes in the classification task.
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_classes = num_classes

    def preprocess_data(self):
        """
        Preprocesses the input data for transfer learning.

        Returns:
            tuple: Tuple containing preprocessed training and test images.
        """
        # Convert DataFrame to NumPy array
        train_data_array = self.train_data.values
        test_data_array = self.test_data.values

        # Reshape the data to (num_samples, height, width, channels)
        train_images = train_data_array.reshape(-1, 5, 8, 1)
        test_images = test_data_array.reshape(-1, 5, 8, 1)

        # Convert the images to black and white (grayscale)
        train_images_bw = np.mean(train_images, axis=3, keepdims=True)
        test_images_bw = np.mean(test_images, axis=3, keepdims=True)

        # Replicate the single channel into three channels
        train_images_rgb = np.concatenate([train_images_bw] * 3, axis=-1)
        test_images_rgb = np.concatenate([test_images_bw] * 3, axis=-1)

        # Pad images with zeros to meet the minimum size requirement
        train_images_padded = self.pad_images(train_images_rgb)
        test_images_padded = self.pad_images(test_images_rgb)

        return train_images_padded, test_images_padded

    def pad_images(self, images):
        """
        Pads input images to meet the minimum size requirement.

        Args:
            images (array-like): Input images.

        Returns:
            array: Padded images.
        """
        padded_images = np.zeros((images.shape[0], 32, 32, 3))
        padded_images[:, 0:5, 0:8, :] = images
        return padded_images

    def transfer_learning(self):
        """
        Performs transfer learning using MobileNetV2.

        Returns:
            Model: The trained transfer learning model.
        """
        # Preprocess the data
        train_images_padded, test_images_padded = self.preprocess_data()

        # Load pre-trained MobileNetV2 model without the top (classification) layer
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(33, 33, 3))

        # Add ZeroPadding2D layer to handle padding
        model = Sequential()
        model.add(ZeroPadding2D(padding=((0, 1), (0, 0))))  # Adjust padding to match MobileNetV2 input size
        model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))  # Adjust padding to match MobileNetV2 input size
        model.add(base_model)

        # Add a global average pooling layer
        model.add(GlobalAveragePooling2D())

        # Add a fully connected layer with softmax activation for classification
        model.add(Dense(self.num_classes, activation='softmax'))

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

        # Train the model with early stopping
        model.fit(train_images_padded, self.train_labels, epochs=50, batch_size=64,
                  validation_data=(test_images_padded, self.test_labels), callbacks=[early_stopping])

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(test_images_padded, self.test_labels, verbose=0)
        print(f'Test accuracy: {test_accuracy}')

        # Save model
        model.save('deep_learning_model.h5')
        return model


class ClusteringModel:
    """
    A class to perform various clustering algorithms and visualize their results.

    Attributes:
        data_train (DataFrame): The training dataset.
        data_test (DataFrame): The testing dataset.
        n_clusters (int): The number of clusters to form.
        train_labels (array): Labels generated by clustering algorithms for training data.
        test_labels (array): Labels generated by clustering algorithms for testing data.

    Methods:
        __init__(self, data_train, data_test, n_clusters):
            Initializes ClusteringModel with the provided datasets and number of clusters.

        hierarchical_clustering(self):
            Performs hierarchical clustering and visualizes the dendrogram.

        k_means(self):
            Performs K-Means clustering and visualizes the clusters.

        gaussian_mixture_model(self):
            Performs Gaussian Mixture Model clustering and visualizes the clusters.

        optics(self):
            Performs OPTICS clustering and visualizes the clusters.

        plot_clusters(self, data, labels):
            Plots clusters in 2D using Principal Component Analysis.

        perform_clustering(self):
            Performs all clustering methods and visualizes their results.

    """

    def __init__(self, data_train, data_test, n_clusters):
        self.data_train = data_train
        self.data_test = data_test
        self.n_clusters = n_clusters
        self.train_labels = None
        self.test_labels = None

    def hierarchical_clustering(self):
        print("Hierarchical Clustering:")
        # Random sample 1% (we have too much data to compute efficiently so we need to reduce it)
        random_sample = self.data_test.sample(n=int(0.01 * len(self.data_test)), replace=False)
        # Plot dendrogram
        sch.dendrogram(sch.linkage(random_sample, method='ward'), color_threshold=30)
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

    def k_means(self):
        print("\nK-Means Clustering:")
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit_predict(self.data_train)
        self.test_labels = kmeans.predict(self.data_test)
        self.plot_clusters(self.data_test, self.test_labels)

    def gaussian_mixture_model(self):
        print("\nGaussian Mixture Model:")
        # Random sample 1% (we have too much data to compute efficiently so we need to reduce it)
        random_sample = self.data_train.sample(n=int(0.01 * len(self.data_train)), replace=False)
        gmm = GaussianMixture(n_components=self.n_clusters)
        gmm.fit_predict(random_sample)
        self.test_labels = gmm.predict(self.data_test)
        self.plot_clusters(self.data_test, self.test_labels)

    def optics(self):
        print("\nOPTICS Clustering:")
        # Random sample 10% (we have too much data to compute efficiently so we need to reduce it)
        random_sample = self.data_test.sample(n=int(0.1 * len(self.data_test)), replace=False)
        optics = OPTICS(min_samples=3)
        self.test_labels = optics.fit(random_sample)
        self.plot_clusters(random_sample, self.test_labels.labels_)

    def plot_clusters(self, data, labels):
        # Perform PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        # Plot clusters in 2D
        sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette='viridis')
        plt.title("Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

    def perform_clustering(self):
        self.hierarchical_clustering()
        self.k_means()
        self.gaussian_mixture_model()
        self.optics()


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

# Divide the data into features and target variable
X = data_loader.data.drop(columns=[data_loader.target])
y = data_loader.data[data_loader.target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model dictionary for the optimization
models_dict = {
    "KNN": {"model": KNeighborsClassifier, "n_neighbors": (3, 5, 7)},
    "LogisticRegression": {"model": LogisticRegression, "C_values": (0.1, 1.0, 10.0), "penalty": (None, 'l2')},
    "DecisionTree": {"model": DecisionTreeClassifier, "max_depth_values": (None, 10, 20)}
}

# Create an instance of ModelBuilder
builder = ModelBuilding(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))

# Build the models
builder.build_models(models_dict)
builder.evaluate_best_model()

# Serialize the builder object
with open('builder.pkl', 'wb') as f:
    pickle.dump(builder, f)

with open('builder.pkl', 'rb') as f:
    builder = pickle.load(f)
print("\nBest model:", builder.best_model_checked)
print("Best model parameters:", builder.best_model_params_checked)

# Perform bagging on the best model
bagging = BaggingClassifier(builder.best_model_checked(**builder.best_model_params_checked), np.array(X_train),
                            np.array(y_train), np.array(X_test), np.array(y_test))
bagging.examine_bagging()
bagging.evaluate()

# Perform AdaBoost on the best model
# adaboost = AdaBoostClassifier(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), builder)
# adaboost.train_adaboost()
# adaboost.evaluate()

tl_model = TransferLearningModel(np.array(X_train), np.array(y_train),
                                 np.array(X_test), np.array(y_test), num_classes=10)
trained_model = tl_model.transfer_learning()

clustering_model = ClusteringModel(np.array(X_train), np.array(X_test), 10)
clustering_model.perform_clustering()
