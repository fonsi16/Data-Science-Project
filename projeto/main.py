#%% 0- Classes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from scipy.stats import entropy, ttest_ind, f_oneway, ttest_rel, wilcoxon, kruskal, friedmanchisquare, probplot, shapiro
from scipy.fftpack import fft
from mrmr import mrmr_classif
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols


class DataLoader:

    def __init__(self, filename, target):
        """
        Initializes the DataLoader with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        self.filename = filename

        self.data = None
        self.labels = None
        self.numerical_features = []
        self.categorical_features = []

        # Load data
        self._load_data(target)

    def _load_data(self, target):
        """
        Loads the dataset from the specified filename,
        splits it into training and testing sets using train_test_split(),
        and assigns the data and labels to the appropriate attributes.
        """
        try:
            # Load the dataset
            self.data = pd.read_csv(self.filename)

            self.data.target = target

            # Validate if the target column exists in the dataset
            if target not in self.data.columns:
                raise ValueError(f"Target column '{target}' not found in the dataset.")

            self.labels = self.data[target]

            # Iterate over columns and categorize them
            for column in self.data.columns:
                if len(self.data[column].unique()) > 2:
                    self.numerical_features.append(column)
                else:
                    self.categorical_features.append(column)

            print("Data loaded successfully.")

        except FileNotFoundError:
            print("File not found. Please check the file path.")


class DataManipulator(DataLoader):

    def __init__(self, filename, target):

        super().__init__(filename, target)

        print("\nData Description:")
        self.describe_variables()

    def describe_variables(self):
        print("\nInformation of Data:")
        print(self.data.info())

        print("\nUnique values of features:")
        print(self.data.nunique())

        print("\nStatistical distribution of each variable:")
        print(self.data.describe())


class DataPreProcessing:

    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader

        self.encode_data()

        self.determine_range()

    def determine_range(self):

        # Display the range of values for each variable without considering the class label
        print("\nRange of values for each variable:")
        print(self.data_loader.data.max() - self.data_loader.data.min())

    def _age_encode(self):
        age = self.data_loader.data["AgeCategory"]
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
        self.data_loader.data["AgeCategory"] = np.select(condition, choice, default=0)

    def _race_encode(self):
        race = self.data_loader.data["Race"]
        condition = [
            race == "White", race == "Black",
            race == "Hispanic", race == "Asian",
            race == "American Indian/Alaskan Native", race == "Other"]
        choice = [1, 2, 3, 4, 5, 6]
        self.data_loader.data["Race"] = np.select(condition, choice)

    def _gen_health_encode(self):
        gen_health = self.data_loader.data["GenHealth"]
        condition = [
            gen_health == "Excellent", gen_health == "Very good",
            gen_health == "Good", gen_health == "Fair",
            gen_health == "Poor"]
        choice = [1, 2, 3, 4, 5]
        self.data_loader.data["GenHealth"] = np.select(condition, choice)

    def encode_data(self):

        # Map categorical features to numerical values
        self.data_loader.data["HeartDisease"] = self.data_loader.data["HeartDisease"].map({"No": 0, "Yes": 1})
        self.data_loader.data["Smoking"] = self.data_loader.data["Smoking"].map({"No": 0, "Yes": 1})
        self.data_loader.data["AlcoholDrinking"] = self.data_loader.data["AlcoholDrinking"].map({"No": 0, "Yes": 1})
        self.data_loader.data["Stroke"] = self.data_loader.data["Stroke"].map({"No": 0, "Yes": 1})
        self.data_loader.data["DiffWalking"] = self.data_loader.data["DiffWalking"].map({"No": 0, "Yes": 1})
        self.data_loader.data["Sex"] = self.data_loader.data["Sex"].map({"Female": 0, "Male": 1})
        self.data_loader.data["Diabetic"] = self.data_loader.data["Diabetic"].map(
            {"No": 0, "No, borderline diabetes": 0, "Yes (during pregnancy)": 1, "Yes": 1})
        self.data_loader.data["PhysicalActivity"] = self.data_loader.data["PhysicalActivity"].map({"No": 0, "Yes": 1})
        self.data_loader.data["Asthma"] = self.data_loader.data["Asthma"].map({"No": 0, "Yes": 1})
        self.data_loader.data["KidneyDisease"] = self.data_loader.data["KidneyDisease"].map({"No": 0, "Yes": 1})
        self.data_loader.data["SkinCancer"] = self.data_loader.data["SkinCancer"].map({"No": 0, "Yes": 1})

        # Encode numerical features
        self._age_encode()
        self._race_encode()
        self._gen_health_encode()

        print("\nProcessed Dataset:")
        print(self.data_loader.data.info())


class DataCleaning:
    """
    Class for cleaning operations.

    Methods:
        remove_duplicates(): Remove duplicate rows from the dataset.
        handle_missing_values(strategy='mean'): Handle missing values using the specified strategy.
        remove_outliers(threshold=3): Remove outliers from the dataset
    """

    def __init__(self, data_loader):

        self.data_loader = data_loader

        print("\nOriginal Dataset before cleaning:")
        print(self.data_loader.data.info())

    def handle_missing_values(self):

        print("Missing values:\n", self.data_loader.data.isnull().sum())

        if self.data_loader.data.isnull().sum().sum() > 0:
            self.data_loader.data = self.data_loader.data.dropna()

    def remove_duplicates(self):

        print("Duplicate Rows:", self.data_loader.data.duplicated().sum())

        if self.data_loader.data.duplicated().sum() > 0:
            self.data_loader.data = self.data_loader.data.drop_duplicates(keep='first')

    def detect_and_remove_outliers(self):

        print("\nDetecting outliers (only numerical values):")
        features_to_delete = []
        for feature in self.data_loader.data:

            # Check if the feature is binary (0 or 1)
            if set(self.data_loader.data[feature]) == {0, 1}:
                # Skip processing binary features
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

    def __init__(self, data_loader):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.data_loader = data_loader

        self.valid_plot_types = ['count', 'hist', 'correlation', 'box', 'barh']

        self.feature_names = self.data_loader.data.columns.tolist()
        self.labels = self.data_loader.data['HeartDisease'].unique().tolist()

    def plot_all_features(self):

        num_features = len(self.feature_names)
        num_cols = 4  # Adjust the number of columns to control subplot arrangement
        num_rows = int(np.ceil(num_features / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        fig.suptitle('All Features', fontsize=20)

        for idx, ax in enumerate(axes.flat):
            if idx < num_features:
                ax.set_title(f'Feature {self.feature_names[idx]}', fontsize=12)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True)

                if self.labels is not None:
                    # Add a plot per feature and label
                    for label in self.labels:
                        mask = np.array(self.data_loader.data['HeartDisease'] == label)
                        ax.hist(self.data_loader.data.loc[mask, self.feature_names[idx]], bins=20, alpha=0.7, label=label)
                    ax.legend()

        plt.tight_layout()
        plt.show()

    def plots(self, plot_types):
        for plot_type in plot_types:
            # Check if the selected plots are in the list of available plots
            if plot_type not in self.valid_plot_types:
                print(
                    f"Ignoring invalid plot type: {plot_type}. Supported plot types: {', '.join(self.valid_plot_types)}")
                continue

            for feature in self.data_loader.data.columns:
                # Create a figure with a single subplot for each feature
                if plot_type == 'count' and feature in self.data_loader.categorical_features:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(x=feature, data=self.data_loader.data, hue=self.data_loader.data.target, ax=ax)
                    ax.set_title(f'Countplot of {feature} by {self.data_loader.data.target}')
                    plt.show()
                if plot_type == 'box' and feature in self.data_loader.numerical_features:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x='AlcoholDrinking', y=feature, data=self.data_loader.data, ax=ax, hue=self.data_loader.data.target)
                    ax.set_title(f'Boxplot of {feature} by {self.data_loader.data.target}')
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
            X = self.data_loader.data.drop(columns=[self.data_loader.data.target])  # Features
            y = self.data_loader.data[self.data_loader.data.target]  # Target variable
            clf.fit(X, y)

            # Calculate permutation importance
            result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            perm_sorted_idx = result.importances_mean.argsort()

            # Visualize feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x=result.importances_mean[perm_sorted_idx], y=X.columns[perm_sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.ylabel('Features')
            plt.title('Permutation Importance')
            plt.show()


class DataAnalysis:
    def __init__(self, data_loader):
        """
        Initialize the DataAnalysis class.

        Parameters:
        - dataset (array-like): The dataset to be analyzed.
        - labels (array-like): The labels corresponding to each of the dataset samples.
        - columns_names (list): List of column (features) names for the dataset.

        This method initializes the DataAnalysis class by setting up the attributes
        for the dataset, labels, and column names.
        """
        self.data_loader = data_loader

        self.valid_plot_types = ['count', 'hist', 'kde', 'correlation', 'box', 'split_violin', 'barh']

    def bmi_class(self):
        bmi = self.df["BMI"]
        condition = [bmi < 16, bmi < 17, bmi < 18.5, bmi < 25, bmi < 30, bmi < 35, bmi < 40, bmi >= 40]
        choice = [1, 2, 3, 4, 5, 6, 7, 8]
        self.df["BMIClass"] = np.select(condition, choice)

    def sleep_class(self):
        sleep = self.df["SleepTime"]
        condition = [sleep < 6, sleep < 9, sleep >= 9]
        choice = [1, 2, 3]
        self.df["SleepClass"] = np.select(condition, choice)

    def badHealth_feature(self):
        smoker = self.df["Smoking"]
        alcohol = self.df["AlcoholDrinking"]
        stroke = self.df["Stroke"]
        diffWalk = self.df["DiffWalking"]
        diabetic = self.df["Diabetic"]
        asthma = self.df["Asthma"]

        condition = (smoker + alcohol + stroke + diffWalk + diabetic + asthma)

        self.df["BadHealthScore"] = condition


class DimensionalityReduction:
    def __init__(self, dataset):
        """
        Initialize the DimensionalityReduction object with the dataset.
        """
        self.dataset = dataset

        # Sample 20% of the data
        self.dataset = self.dataset.sample(frac=0.2, random_state=42)

        self.data = StandardScaler().fit_transform(self.dataset.drop(columns=['HeartDisease']))
        self.target = self.dataset['HeartDisease']

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.

        Parameters:
        - projection: The projected data.
        - title: The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        if projection.shape[1] == 1:
            plt.scatter(projection, np.zeros_like(projection), c=self.target, alpha=0.5)
        else:
            plt.scatter(projection[:, 0], projection[:, 1], c=self.target, alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_lda(self, n_components=3):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.target)

    def compute_tsne(self, n_components=2, perplexity=3):
        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between embedded points.
        - metric: The distance metric to use.

        Returns:
        - umap_projection: The projected data using UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.

        Returns:
        - lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self.data)


class HypothesisTester:

    def unpaired_t_test(self, group1, group2):
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

    def unpaired_anova(self, *groups):
        """
        Perform unpaired ANOVA for more than two groups.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        f_statistic, p_value = f_oneway(*groups)
        return f_statistic, p_value

    def paired_t_test(self, group1, group2):
        """
        Perform paired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.
                  Should have the same length as group1.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_rel(group1, group2)
        return t_statistic, p_value

    def paired_anova(self, data):
        """
        Perform paired ANOVA (repeated measures ANOVA) for more than two groups.

        Parameters:
        - data: Pandas DataFrame containing the data with columns representing different conditions.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        model = ols('value ~ C(condition)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table['F'][0], anova_table['PR(>F)'][0]

    def wilcoxon_ranksum_test(self, group1, group2):
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

    def wilcoxon_signedrank_test(self, group1, group2):
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Defines the alternative hypothesis with ‘greater’ option, this the distribution underlying d is stochastically
        greater than a distribution symmetric about zero; d represent the difference between the paired samples:
        d = x - y if both x and y are provided, or d = x otherwise.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.
                  Should have the same length as group1.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = wilcoxon(group1, group2, alternative="greater")
        return statistic, p_value

    def kruskal_wallis_test(self, *groups):
        """
        Perform Kruskal-Wallis H test for independent samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = kruskal(*groups)
        return statistic, p_value

    def friedman_test(self, *groups):
        """
        Perform Friedman test for related samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object representing measurements of the same individuals under different conditions.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = friedmanchisquare(*groups)
        return statistic, p_value

    def qq_plots(self, variable_names, *data_samples, distribution='norm'):
        """
        Generate Q-Q plots for multiple data samples.

        Parameters:
        - variable_names: List with the names of the variables to be plotted
        - data_samples: Variable number of 1D array-like objects representing the data samples.
        - distribution: String indicating the theoretical distribution to compare against. Default is 'norm' for normal
        distribution.

        Returns:
        - None (displays the Q-Q plots)
        """
        num_samples = len(data_samples)
        num_rows = (num_samples + 1) // 2  # Calculate the number of rows for subplots
        num_cols = 2 if num_samples > 1 else 1  # Ensure at least 1 column for subplots

        # Adjust the height of the figure to fit all Q-Q plots without overlapping
        fig_height = 6 * num_rows  # Adjust this value as needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, fig_height))
        axes = axes.flatten()  # Flatten axes if multiple subplots

        for i, data in enumerate(data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f'Q-Q Plot ({distribution})')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel(variable_names[i])

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()

    def test_normality(self, variable_names, *data_samples):
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
        for name, data in zip(variable_names, data_samples):
            results[name] = shapiro(data)
        for variable_name, shapiro_result in results.items():
            print(f'{variable_name}:')
            print(f'Shapiro-Wilk test - Test statistic: {shapiro_result.statistic}, p-value: {shapiro_result.pvalue}')
        return results


class FeatureExtractor:
    """
    A class to extract various types of features from a dataset.
    """

    def __init__(self, dataset):
        """
        Initializes the FeatureExtractor object.

        Parameters:
        - data (numpy.ndarray): The input dataset.
        """
        self.data = dataset

        self.feature_names = self.data.columns.tolist()
        self.labels = self.data['HeartDisease'].unique().tolist()

        self.all_features = []

    def _statistical_features(self):
        """
        Computes statistical features per sample (row) of the dataset.

        Returns:
        - numpy.ndarray: An array containing statistical features per sample.
        """
        # Compute statistical features
        mean = np.mean(self.data, axis=1)
        std_dev = np.std(self.data, axis=1)
        median = np.median(self.data, axis=1)
        min_val = np.min(self.data, axis=1)
        max_val = np.max(self.data, axis=1)
        # Append feature names
        self.feature_names.extend(['Mean'])
        self.feature_names.extend(['Std_Dev'])
        self.feature_names.extend(['Median'])
        self.feature_names.extend(['Min'])
        self.feature_names.extend(['Max'])
        return np.column_stack((mean, std_dev, median, min_val, max_val))

    def _pairwise_differences(self):
        num_features = self.data.shape[1]
        pairwise_diff_df = pd.DataFrame()

        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                feature_name = f'pairwise_diff_{i + 1}_vs_{j + 1}'
                pairwise_diff_df[feature_name] = np.abs(self.data.iloc[:, i] - self.data.iloc[:, j])
                self.feature_names.append(feature_name)

        return pairwise_diff_df

    def _frequency_domain_features(self):
        """
        Computes frequency domain features per sample (row) of the dataset using FFT.

        Returns:
        - numpy.ndarray: An array containing frequency domain features per sample.
        """
        # Compute frequency domain features using FFT
        fft_result = fft(self.data)
        # Append feature names
        self.feature_names.extend(['FFT'])
        return np.abs(fft_result).mean(axis=1).reshape(-1, 1)

    def _entropy_features(self):

        # Compute entropy-based features for each row
        entropy_vals = np.array([entropy(row) for idx, row in self.data.iterrows()])

        # Append feature names
        self.feature_names.append('Entropy')

        return entropy_vals.reshape(-1, 1)

    def extract_features(self):
        """
        Extracts various types of features from the dataset.

        Returns:
        - pandas.DataFrame: A dataframe containing all extracted features.
        """
        # Extract and combine all features with the original features passed in data
        self.all_features = np.hstack((self.data, self._statistical_features(), self._pairwise_differences(),
                                  self._frequency_domain_features(), self._entropy_features()))
        # Create pandas dataframe
        return pd.DataFrame(data=self.all_features, columns=self.feature_names)

    def plot_all_features(self):
        """
        Plot histograms for all extracted features.
        Each histogram is plotted separately for each feature, with optional coloring based on provided labels.
        """
        num_features = self.all_features.shape[1]
        num_cols = 4  # Adjust the number of columns to control subplot arrangement
        num_rows = int(np.ceil(num_features / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        fig.suptitle('All Features', fontsize=20)

        for idx, ax in enumerate(axes.flat):
            if idx < num_features:
                ax.set_title(f'Feature {self.feature_names[idx]}', fontsize=12)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True)

                if self.labels is not None:
                    # Add a plot per feature and label
                    for label in self.labels:
                        mask = np.array(self.data['HeartDisease'] == label)
                        ax.hist(self.all_features[mask, idx], bins=20, alpha=0.7, label=label)
                    ax.legend()

        plt.tight_layout()
        plt.show()


class FeatureSelector:
    def __init__(self, data, labels):
        """
        Initialize the FeatureSelector instance.

        Parameters:
        - data (numpy.ndarray): The input data array with shape (n_samples, n_features).
        - labels (numpy.ndarray): The labels array with shape (n_samples,).
        """
        self.data = data
        self.labels = labels

        self._sanity_check()

    def _sanity_check(self):
        # Perform basic sanity checks
        print("Dataset Shape:", self.data.shape)

        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())

        # Check data types
        print("\nData Types:")
        print(self.data.dtypes)

        # Check for zero variance features
        zero_variance_features = self.data.columns[self.data.var() == 0]
        print("\nZero Variance Features:", zero_variance_features)

    def select_features_mrmr(self, k=5):
        """
        Select features using mRMR (minimum Redundancy Maximum Relevance).

        Parameters:
        - k (int): The number of features to select. Default is 5.

        Returns:
        - List: The selected features as a list.
        """
        # Return the selected features
        return mrmr_classif(X=self.data, y=self.labels, K=k)


#%% 1- Pre Processing and EDA

path = 'data/heart_2020.csv'

data_loader = DataManipulator(path, 'HeartDisease')

# Process all the data to numeric values and determine the range of values for each variable
data_preprocessing = DataPreProcessing(data_loader)

# Visualization of the outliers and all the histograms
data_visualization = DataVisualization(data_loader)
data_visualization.plot_all_features()
data_visualization.plots(['box'])

#data_cleaner = DataCleaning(data_loader)
# Verify the presence of missing values, duplicated data and outliers and clean the data
#data_cleaner.handle_missing_values()
#data_cleaner.remove_duplicates()
#data_cleaner.detect_and_remove_outliers()

print("\nCleansed Dataset:")
print(data_loader.data.info)

# Save the cleaned dataset to a new csv file
data_loader.data.to_csv('data/heart_2020_cleaned.csv', index=False)

#%% 2 - EDA

data_visualization_cleaned = DataVisualization(data_loader)
data_visualization_cleaned.plots(['correlation', 'barh'])

# # Plots after the cleansing and processing
# data_analysis_instance.plots(['correlation', 'barh'])
#
# # Initialize DimensionalityReduction object with the dataset
# dr = DimensionalityReduction(dataset_cleaned)
#
# # Compute and plot PCA projection
# dr.plot_projection(dr.compute_pca(), 'PCA Projection')
# # Compute and plot t-SNE projection
# dr.plot_projection(dr.compute_tsne(), 't-SNE Projection')
# # Compute and plot UMAP projection
# dr.plot_projection(dr.compute_umap(), 'UMAP Projection')


#%% 2- Hypothesis Testing

# BMI = dataset_cleaned['BMI']
# Smoke = dataset_cleaned['Smoking']
# Alcohol = dataset_cleaned['AlcoholDrinking']
# Stroke = dataset_cleaned['Stroke']
# PH = dataset_cleaned['PhysicalHealth']
# MH = dataset_cleaned['MentalHealth']
# DW = dataset_cleaned['DiffWalking']
# Sex = dataset_cleaned['Sex']
# AC = dataset_cleaned['AgeCategory']
# Diabetic = dataset_cleaned['Diabetic']
# PA = dataset_cleaned['PhysicalActivity']
# GH = dataset_cleaned['GenHealth']
# ST = dataset_cleaned['SleepTime']
# Asthma = dataset_cleaned['Asthma']
# KD = dataset_cleaned['KidneyDisease']
# SC = dataset_cleaned['SkinCancer']
#
# # Column Data with Hearth Disease
# BMI_with_HD = dataset_cleaned['BMI'][dataset_cleaned['HeartDisease'] == 1]
# Smoke_with_HD = dataset_cleaned['Smoking'][dataset_cleaned['HeartDisease'] == 1]
# Alcohol_with_HD = dataset_cleaned['AlcoholDrinking'][dataset_cleaned['HeartDisease'] == 1]
# Stroke_with_HD = dataset_cleaned['Stroke'][dataset_cleaned['HeartDisease'] == 1]
# PH_with_HD = dataset_cleaned['PhysicalHealth'][dataset_cleaned['HeartDisease'] == 1]
# MH_with_HD = dataset_cleaned['MentalHealth'][dataset_cleaned['HeartDisease'] == 1]
# DW_with_HD = dataset_cleaned['DiffWalking'][dataset_cleaned['HeartDisease'] == 1]
# Sex_with_HD = dataset_cleaned['Sex'][dataset_cleaned['HeartDisease'] == 1]
# AC_with_HD = dataset_cleaned['AgeCategory'][dataset_cleaned['HeartDisease'] == 1]
# Diabetic_with_HD = dataset_cleaned['Diabetic'][dataset_cleaned['HeartDisease'] == 1]
# PA_with_HD = dataset_cleaned['PhysicalActivity'][dataset_cleaned['HeartDisease'] == 1]
# GH_with_HD = dataset_cleaned['GenHealth'][dataset_cleaned['HeartDisease'] == 1]
# ST_with_HD = dataset_cleaned['SleepTime'][dataset_cleaned['HeartDisease'] == 1]
# Asthma_with_HD = dataset_cleaned['Asthma'][dataset_cleaned['HeartDisease'] == 1]
# KD_with_HD = dataset_cleaned['KidneyDisease'][dataset_cleaned['HeartDisease'] == 1]
# SC_with_HD = dataset_cleaned['SkinCancer'][dataset_cleaned['HeartDisease'] == 1]
#
# With_HD = [BMI_with_HD, Smoke_with_HD, Alcohol_with_HD, Stroke_with_HD, PH_with_HD, MH_with_HD, DW_with_HD,
#            Sex_with_HD, AC_with_HD, Diabetic_with_HD, PA_with_HD, GH_with_HD, ST_with_HD,
#            Asthma_with_HD, KD_with_HD, SC_with_HD]
#
# # Column Data without Hearth Disease
# BMI_without_HD = dataset_cleaned['BMI'][dataset_cleaned['HeartDisease'] == 0]
# Smoke_without_HD = dataset_cleaned['Smoking'][dataset_cleaned['HeartDisease'] == 0]
# Alcohol_without_HD = dataset_cleaned['AlcoholDrinking'][dataset_cleaned['HeartDisease'] == 0]
# Stroke_without_HD = dataset_cleaned['Stroke'][dataset_cleaned['HeartDisease'] == 0]
# PH_without_HD = dataset_cleaned['PhysicalHealth'][dataset_cleaned['HeartDisease'] == 0]
# MH_without_HD = dataset_cleaned['MentalHealth'][dataset_cleaned['HeartDisease'] == 0]
# DW_without_HD = dataset_cleaned['DiffWalking'][dataset_cleaned['HeartDisease'] == 0]
# Sex_without_HD = dataset_cleaned['Sex'][dataset_cleaned['HeartDisease'] == 0]
# AC_without_HD = dataset_cleaned['AgeCategory'][dataset_cleaned['HeartDisease'] == 0]
# Diabetic_without_HD = dataset_cleaned['Diabetic'][dataset_cleaned['HeartDisease'] == 0]
# PA_without_HD = dataset_cleaned['PhysicalActivity'][dataset_cleaned['HeartDisease'] == 0]
# GH_without_HD = dataset_cleaned['GenHealth'][dataset_cleaned['HeartDisease'] == 0]
# ST_without_HD = dataset_cleaned['SleepTime'][dataset_cleaned['HeartDisease'] == 0]
# Asthma_without_HD = dataset_cleaned['Asthma'][dataset_cleaned['HeartDisease'] == 0]
# KD_without_HD = dataset_cleaned['KidneyDisease'][dataset_cleaned['HeartDisease'] == 0]
# SC_without_HD = dataset_cleaned['SkinCancer'][dataset_cleaned['HeartDisease'] == 0]
#
# Without_HD = [BMI_without_HD, Smoke_without_HD, Alcohol_without_HD, Stroke_without_HD, PH_without_HD,
#               MH_without_HD, DW_without_HD, Sex_without_HD, AC_without_HD, Diabetic_without_HD,
#               PA_without_HD, GH_without_HD, ST_without_HD, Asthma_without_HD, KD_without_HD, SC_without_HD]
#
# # Initialize the HypothesisTester class with the data
# tester = HypothesisTester()
#
# # Perform normality analysis, first by visual checking using a Q-Q plot and then by normality test
# tester.qq_plots(['BMI_with_HD', 'Smoke_with_HD', 'Alcohol_with_HD', 'Stroke_with_HD', 'PH_with_HD', 'MH_with_HD', 'DW_with_HD',
#                 'Sex_with_HD', 'AC_with_HD', 'Diabetic_with_HD', 'PA_with_HD', 'GH_with_HD', 'ST_with_HD',
#                 'Asthma_with_HD', 'KD_with_HD', 'SC_with_HD', 'BMI_without_HD', 'Smoke_without_HD', 'Alcohol_without_HD',
#                 'Stroke_without_HD', 'PH_without_HD', 'MH_without_HD', 'DW_without_HD', 'Sex_without_HD', 'AC_without_HD',
#                 'Diabetic_without_HD', 'PA_without_HD', 'GH_without_HD', 'ST_without_HD',
#                 'Asthma_without_HD', 'KD_without_HD', 'SC_without_HD'], BMI_with_HD, Smoke_with_HD, Alcohol_with_HD,
#                 Stroke_with_HD, PH_with_HD, MH_with_HD, DW_with_HD,
#                 Sex_with_HD, AC_with_HD, Diabetic_with_HD, PA_with_HD, GH_with_HD, ST_with_HD,
#                 Asthma_with_HD, KD_with_HD, SC_with_HD, BMI_without_HD, Smoke_without_HD, Alcohol_without_HD,
#                 Stroke_without_HD, PH_without_HD, MH_without_HD, DW_without_HD, Sex_without_HD, AC_without_HD,
#                 Diabetic_without_HD, PA_without_HD, GH_without_HD, ST_without_HD,
#                 Asthma_without_HD, KD_without_HD, SC_without_HD)
#
# tester.test_normality(['BMI_with_HD', 'Smoke_with_HD', 'Alcohol_with_HD', 'Stroke_with_HD', 'PH_with_HD', 'MH_with_HD', 'DW_with_HD',
#                 'Sex_with_HD', 'AC_with_HD','Diabetic_with_HD', 'PA_with_HD', 'GH_with_HD', 'ST_with_HD',
#                 'Asthma_with_HD', 'KD_with_HD', 'SC_with_HD', 'BMI_without_HD', 'Smoke_without_HD', 'Alcohol_without_HD',
#                 'Stroke_without_HD', 'PH_without_HD', 'MH_without_HD', 'DW_without_HD', 'Sex_without_HD', 'AC_without_HD',
#                 'Diabetic_without_HD', 'PA_without_HD', 'GH_without_HD', 'ST_without_HD',
#                 'Asthma_without_HD', 'KD_without_HD', 'SC_without_HD'], BMI_with_HD, Smoke_with_HD, Alcohol_with_HD,
#                 Stroke_with_HD, PH_with_HD, MH_with_HD, DW_with_HD,
#                 Sex_with_HD, AC_with_HD, Diabetic_with_HD, PA_with_HD, GH_with_HD, ST_with_HD,
#                 Asthma_with_HD, KD_with_HD, SC_with_HD, BMI_without_HD, Smoke_without_HD, Alcohol_without_HD,
#                 Stroke_without_HD, PH_without_HD, MH_without_HD, DW_without_HD, Sex_without_HD, AC_without_HD,
#                 Diabetic_without_HD, PA_without_HD, GH_without_HD, ST_without_HD,
#                 Asthma_without_HD, KD_without_HD, SC_without_HD)
#
# #------------------------------TESTES DE ARRAYS DA MESMA FEATURE COM E SEM DOENÇA CARDIACA------------------------------
#
# # Iterate over the indices of the arrays
# for i in range(len(With_HD)):
#
#     # Perform ANOVA test
#     f_stat, p_val_anova = tester.unpaired_anova(With_HD[i], Without_HD[i])
#
#     # Print the results
#     print(f"\nUnpaired ANOVA between the array of {With_HD[i].name} with HeartDisease and the array without : ")
#     print("F-statistic:", f_stat)
#     print("p-value:", p_val_anova)
#
# # Iterate over the indices of the arrays
# for i in range(len(With_HD)):
#
#     # Perform Wilcoxon rank-sum test
#     statistic, p_value = tester.wilcoxon_ranksum_test(With_HD[i], Without_HD[i])
#
#     # Print the results
#     print(f"\nWilcoxon rank-sum test between the array of {With_HD[i].name} with HeartDisease and the array without : ")
#     print("Test statistic:", statistic)
#     print("p-value:", p_value)
#
#
# # Iterate over the indices of the arrays
# for i in range(len(With_HD)):
#
#     # Perform Kruskal-Wallis test
#     statistic, p_value = tester.kruskal_wallis_test(With_HD[i], Without_HD[i])
#
#     # Print the results
#     print(f"\nKruskal-Wallis test between the array of {With_HD[i].name} with HeartDisease and the array without:")
#     print("Test statistic:", statistic)
#     print("p-value:", p_value)

#%% 3- Modeling

# extractor = FeatureExtractor(dataset_cleaned)
#
# # Extract features
# feature_df = extractor.extract_features()
# # Display the dataframe
# print(feature_df)
# # Plot the features per class
# extractor.plot_all_features()
#
# feature_data = feature_df.drop(columns=['HeartDisease'])
# feature_labels = feature_df['HeartDisease']
#
# feature_selector = FeatureSelector(feature_data, feature_labels)
# # Use the select_features_mrmr method
# selected_features_mrmr = feature_selector.select_features_mrmr()
# print("Selected features (mRMR):", selected_features_mrmr)
