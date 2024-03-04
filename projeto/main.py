#%% 0- Classes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from scipy.stats import entropy
from scipy.fftpack import fft
from mrmr import mrmr_classif
from sklearn.feature_selection import SequentialFeatureSelector


class DataAnalysis:
    def __init__(self, dataset_path, target):
        """
        Initialize the DataAnalysis class.

        Parameters:
        - dataset (array-like): The dataset to be analyzed.
        - labels (array-like): The labels corresponding to each of the dataset samples.
        - columns_names (list): List of column (features) names for the dataset.

        This method initializes the DataAnalysis class by setting up the attributes
        for the dataset, labels, and column names.
        """
        self.df = pd.read_csv(dataset_path)
        self.target = target
        self.numerical_features = None
        self.categorical_features = None

        # Validate if the target column exists in the dataset
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the dataset.")

        # Extract the labels from the target column
        self.labels = self.df[self.target]

        # Concatenate the labels to the dataset
        self.df_with_labels = pd.concat([self.df, self.labels], axis=1)

        self.valid_plot_types = ['count', 'hist', 'kde', 'correlation', 'box', 'split_violin']

    def describe_variables(self):
        print("\nInformation of Data:")
        print(self.df.info())

        print("\nStatistical distribution of each variable:")
        print(self.df.describe())

    def determine_range(self):

        # Display the range of values for each variable without considering the class label
        print("\nRange of values for each variable:")
        print(self.df.max() - self.df.min())

        # Display the range of values for each variable per class label
        print("\nRange of values for each variable per class label:")
        #print(self.df_with_labels.max() - self.df_with_labels.min())

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

    def age_category(self):
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

    def race_category(self):
        race = self.df["Race"]
        condition = [
            race == "White", race == "Black",
            race == "Hispanic", race == "Asian",
            race == "American Indian/Alaskan Native", race == "Other"]
        choice = [1, 2, 3, 4, 5, 6]
        self.df["Race"] = np.select(condition, choice)

    def genHealth_category(self):
        genHealth = self.df["GenHealth"]
        condition = [
            genHealth == "Excellent", genHealth == "Very good",
            genHealth == "Good", genHealth == "Fair",
            genHealth == "Poor"]
        choice = [1, 2, 3, 4, 5]
        self.df["GenHealth"] = np.select(condition, choice)

    def badHealth_feature(self):
        smoker = self.df["Smoking"]
        alcohol = self.df["AlcoholDrinking"]
        stroke = self.df["Stroke"]
        diffWalk = self.df["DiffWalking"]
        diabetic = self.df["Diabetic"]
        asthma = self.df["Asthma"]

        condition = (smoker + alcohol + stroke + diffWalk + diabetic + asthma)

        self.df["BadHealthScore"] = condition



    def process_data(self):

        # Turn all the features to numerical values:
        # HeartDisease (0-No / 1-Yes)
        # Smoking (0-No / 1-Yes)
        # AlcoholDrinking (0-No / 1-Yes)
        # Stroke (0-No / 1-Yes)
        # PhysicalHealth - Doesn't need it
        # MentalHealth - Doesn't need it
        # DiffWalking (0-No / 1-Yes)
        # Sex (0-Female / 1-Male)
        # AgeCategory (1-(18-24) / 2-(25-29) / 3-(30-34) / 4-(35-39) / 5-(40-44) / 6-(45-49) / 7-(50-54) / 8-(55-59) / 9-(60-64) / 10-(65-69) / 11-(70-74) / 12-(75-79) / 13-(80 or older))
        # Race (1-White / 2-Black / 3-Hispanic / 4-Asian / 5-American Indian/Alaskan Native / 6-Other)
        # Diabetic (0-No / 0-No, borderline diabetes / 2-Yes (during pregnancy) / 2-Yes)
        # PhysicalActivity (0-No / 1-Yes)
        # GenHealth - (1-Excellent / 2-Very good / 3-Good / 4-Fair / 5-Poor)
        # Asthma (0-No / 1-Yes)
        # KidneyDisease (0-No / 1-Yes)
        # SkinCancer (0-No / 1-Yes)

        # BMIClass (1-<16, 2-<17, 3-<18.5, 4-<25, 5-<30, 6-<35, 7-<40, 8->=40)
        # SleepClass (1-<6, 2-<9, 3->=9)

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

        # Process features
        self.age_category()
        self.race_category()
        self.genHealth_category()

        self.bmi_class()
        self.sleep_class()
        self.badHealth_feature()

        self.numerical_features = ["AgeCategory", "Race", "GenHealth",
                                   "BMI", "BMIClass", "SleepTime",
                                   "SleepClass", "BadHealthScore"]
        self.categorical_features = ["Smoking", "AlcoholDrinking", "Stroke",
                                     "DiffWalking", "Sex", "Diabetic",
                                     "PhysicalActivity", "Asthma",
                                     "KidneyDisease", "SkinCancer"]

        print("\nProcessed Dataset:")
        print(self.df.info())

    def assess_quality(self):

        print("\nOriginal Dataset:")
        print(self.df.info)

        # Original Data Plots
        #data_analysis_instance.plots(['kde'])
        data_analysis_instance.plots(['correlation'])

        print("Missing values:\n", self.df.isnull().sum())
        print("Duplicate Rows:", self.df.duplicated().sum())

        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates(keep='first')

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
                if plot_type == 'count' and feature in self.categorical_features:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(x=feature, data=self.df, hue=self.target, ax=ax)
                    ax.set_title(f'Countplot of {feature} by {self.target}')
                    plt.show()
                if plot_type == 'hist':
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(x=feature, data=self.df, hue=self.target, ax=ax)
                    ax.set_title(f'Histogram of {feature}')
                    plt.show()
                if plot_type == 'kde' and feature in self.numerical_features:
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
                if plot_type == 'box' and feature in self.numerical_features:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=self.target, y=feature, data=self.df, ax=ax)
                    ax.set_title(f'Boxplot of {feature} by {self.target}')
                    plt.show()
                if plot_type == 'split_violin' and feature in self.numerical_features:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.violinplot(x=self.target, y=feature, hue=self.target, split=True, data=self.df, ax=ax)
                    ax.set_title(f'Split Violin Plot of {feature} by {self.target}')
                    plt.show()

        if 'correlation' in plot_types:
            correlation = self.df.corr().round(2)
            plt.figure(figsize=(15, 12))
            sns.heatmap(correlation, annot=True, cmap='YlOrBr', annot_kws={'size': 8})
            plt.title('Correlation Heatmap')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


class PCAAnalysis:
    def __init__(self, dataset_path, num_components):

        # Extract features (X) and target variable (y) from the dataset
        self.X = pd.read_csv(dataset_path)
        self.y = self.X['HeartDisease']

        if num_components > self.X.shape[1]:
            raise ValueError("Number of components cannot exceed the number of features.")

        # Configuration
        self.num_components = num_components

        # Plots
        self.fig, self.axes = None, None

        # Implemented PCA
        self.X_standardized = self._standardize_data()
        self.cov_matrix = self._compute_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self._compute_eigenvalues_eigenvectors()
        self.eigenvalues, self.eigenvectors = self._sort_eigenvectors()
        self.pca_projection = self._project_data()

        # Library PCA
        self.sklearn_pca_projection, self.sklearn_pca = self._apply_sklearn_pca()

    def _standardize_data(self):
        """
        Step 1: Standardize the dataset.
        """
        return StandardScaler().fit_transform(self.X)

    def _compute_covariance_matrix(self):
        """
        Step 2: Compute the covariance matrix.
        """
        return np.cov(self.X_standardized.T)

    def _compute_eigenvalues_eigenvectors(self):
        """
        Step 3: Compute the eigenvectors and eigenvalues.
        """
        return np.linalg.eig(self.cov_matrix)

    def _sort_eigenvectors(self):
        """
        Step 4: Sort eigenvectors based on eigenvalues.
        """
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        return self.eigenvalues[sorted_indices], self.eigenvectors[:, sorted_indices]

    def _project_data(self):
        """
        Step 5: Select the number of principal components and project the data onto them.
        """
        return self.X_standardized.dot(self.eigenvectors[:, :self.num_components])

    def _apply_sklearn_pca(self):
        """
        Apply PCA using sklearn (for comparison).
        """
        pca = PCA(n_components=self.num_components)
        return pca.fit_transform(self.X_standardized), pca

    def display_feature_contributions(self):
        """
        Display feature contributions to principal components.
        """
        print("\nFeature Contributions to Principal Components:")
        for i, eigenvector in enumerate(self.eigenvectors.T):
            print(f"Principal Component {i + 1}:")
            for j, feature_contribution in enumerate(eigenvector):
                print(f"   Feature {j + 1}: {feature_contribution:.4f}")

    def calculate_explained_variance_ratio(self):
        """
        Calculate explained variance ratio for both developed and library PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        print(f"\nExplained Variance of the developed PCA using {self.num_components} component(s): ",
              np.sum(explained_variance_ratio))
        print(f"Explained Variance of the library PCA using {self.num_components} component(s): ",
              np.sum(self.sklearn_pca.explained_variance_ratio_))

    def plot_explained_variance_ratio(self):
        """
        Plot the explained variance ratio of the developed PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(1, self.num_components + 1), explained_variance_ratio, alpha=0.5, align='center')
        for bar, value in zip(bars, explained_variance_ratio):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center',
                     va='bottom')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance Ratio per Principal Component')
        plt.grid(True)
        plt.show()

    def print_pca_projection(self):
        """
        Show the first lines of the developed and the library PCA projection.
        """
        print("\nPCA Projection (Manual):\n", self.pca_projection[:5])
        print("\nPCA Projection (Sklearn):\n", self.sklearn_pca_projection[:5])

    def plot_pca_projections(self):
        """
        Plot PCA projections for principal and for the two principal components in a 4 by 4 grid.
        """

        # For the developed PCA
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.axes[0, 0].scatter(self.pca_projection[:, 0], self.pca_projection[:, 0], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 0].set_title('PCA Projection of the First Principal Component (Manual)')
        self.axes[0, 0].set_xlabel('Principal Component 1')
        self.axes[0, 0].set_ylabel('Principal Component 1')
        self.axes[0, 0].grid(True)

        self.axes[0, 1].scatter(self.pca_projection[:, 0], self.pca_projection[:, 1], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 1].set_title('PCA Projection of the First Two Principal Components (Manual)')
        self.axes[0, 1].set_xlabel('Principal Component 1')
        self.axes[0, 1].set_ylabel('Principal Component 2')
        self.axes[0, 1].grid(True)

        # For th library PCA
        self.axes[1, 0].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 0], c=self.y,
                                cmap='viridis', alpha=0.8)
        self.axes[1, 0].set_title('PCA Projection of the First Principal Component (Sklearn)')
        self.axes[1, 0].set_xlabel('Principal Component 1')
        self.axes[1, 0].set_ylabel('Principal Component 1')
        self.axes[1, 0].grid(True)

        scatter = self.axes[1, 1].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 1],
                                          c=self.y, cmap='viridis', alpha=0.8)
        self.axes[1, 1].set_title('PCA Projection of the First Two Principal Components (Sklearn)')
        self.axes[1, 1].set_xlabel('Principal Component 1')
        self.axes[1, 1].set_ylabel('Principal Component 2')
        self.axes[1, 1].grid(True)

        # The * before scatter.legend_elements() is the unpacking operator in Python, when used before an iterable
        # (such as a list or a tuple), it unpacks the elements of the iterable into positional arguments of a function
        # or method call. In this specific context, scatter.legend_elements() returns a tuple containing two elements:
        # handles and labels. The handles represent the plotted elements (in this case, the points in the scatter plot),
        # and the labels represent the corresponding labels for those elements (in this case, the class labels). By
        # using * before scatter.legend_elements(), we are unpacking the tuple returned by scatter.legend_elements()
        # into separate arguments, which are then passed as positional arguments to the legend() method of the
        # matplotlib.axes.Axes object.
        self.axes[1, 1].add_artist(
            self.axes[1, 1].legend(*scatter.legend_elements(), title="Classes", loc="lower right"))
        plt.tight_layout()
        plt.show()


class DimensionalityReduction:
    def __init__(self, dataset_path):
        """
        Initialize the DimensionalityReduction object with the dataset.
        """
        self.dataset = pd.read_csv(dataset_path)

        # Sample 5% of the data
        self.dataset = self.dataset.sample(frac=0.05, random_state=42)

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


class FeatureExtractor:
    """
    A class to extract various types of features from a dataset.
    """

    def __init__(self, dataset_path):
        """
        Initializes the FeatureExtractor object.

        Parameters:
        - data (numpy.ndarray): The input dataset.
        """
        self.data = pd.read_csv(dataset_path)

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
        """
        Computes the pairwise absolute differences between each pair of features per line.

        Returns:
        -------
        pd.DataFrame:
            A DataFrame containing pairwise absolute differences between each pair of features.
            The DataFrame has (n-1) * n / 2 columns, where n is the number of features in the dataset.
            Each row represents the absolute differences between pairs of features for a single data point.
        """
        # Calculate the number of features
        num_features = self.data.shape[1]

        # Initialize an empty DataFrame to store the pairwise differences
        pairwise_diff_df = pd.DataFrame()

        # Compute pairwise absolute differences for each pair of features
        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                feature_name = f'pairwise_diff_{i + 1}_vs_{j + 1}'
                pairwise_diff_df[feature_name] = np.abs(self.data[:, i] - self.data[:, j])
                self.feature_names.extend([feature_name])

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
        """
        Computes entropy-based features per sample (row) of the dataset.

        Returns:
        - numpy.ndarray: An array containing entropy-based features per sample.
        """
        # Compute entropy-based features
        entropy_vals = [entropy(self.data[i]) for i in range(len(self.data))]
        # Append feature names
        self.feature_names.extend(['Entropy'])
        return np.array(entropy_vals).reshape(-1, 1)

    def _area_based_features(self):
        """
        Calculate area-based features including petal area, sepal area, and petal to sepal area ratio.

        Returns:
        - numpy.ndarray: An array containing the calculated features.
        """
        petal_length = self.data[:, 2]
        petal_width = self.data[:, 3]
        sepal_length = self.data[:, 0]
        sepal_width = self.data[:, 1]
        # Calculate petal area (assuming petal shape is close to an ellipse)
        petal_area = np.pi * (petal_length / 2) * (petal_width / 2)
        # Calculate sepal area (assuming sepal shape is close to a rectangle)
        sepal_area = sepal_length * sepal_width
        # Calculate petal to sepal area ratio
        ratio_petal_sepal_area = petal_area / sepal_area
        # Add the features names
        self.feature_names.extend(['Petal_area'])
        self.feature_names.extend(['Sepal_area'])
        self.feature_names.extend(['Ratio_petal_sepal_area'])
        # Stack the calculated features horizontally
        return np.column_stack((petal_area, sepal_area, ratio_petal_sepal_area))

    def extract_features(self):
        """
        Extracts various types of features from the dataset.

        Returns:
        - pandas.DataFrame: A dataframe containing all extracted features.
        """
        # Extract and combine all features with the original features passed in data
        self.all_features = np.hstack((self.data, self._statistical_features(), self._pairwise_differences(),
                                  self._frequency_domain_features(), self._entropy_features(), self._area_based_features()))
        # Create pandas dataframe
        return pd.DataFrame(data=self.all_features, columns=self.feature_names)

    def plot_all_features(self):
        """
        Plot histograms for all extracted features.
        Each histogram is plotted separately for each feature, with optional coloring based on provided labels.
        """
        num_features = self.all_features.shape[1]
        num_rows = int(np.ceil(np.sqrt(num_features)))
        num_cols = int(np.ceil(num_features / num_rows))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        fig.suptitle('All Features', fontsize=20)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_features:
                    ax = axes[i, j]
                    ax.set_title(f'Feature {self.feature_names[idx]}', fontsize=12)
                    ax.set_xlabel('Value', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.grid(True)
                    if self.labels is not None:
                        # Add a plot per feature
                        unique_labels = np.unique(self.labels)
                        for label in unique_labels:
                            ax.hist(self.all_features[self.labels == label, idx], bins=20, alpha=0.7, label=label)
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

    def select_features_sequential(self, k=5):
        """
        Select features using sequential feature selection with LDA as the classifier.

        Parameters:
        - k (int): The number of features to select. Default is 5.

        Returns:
        - numpy.ndarray: The selected features array with shape (n_samples, k).
        """
        # Sequential forward feature selection
        sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), n_features_to_select=k, direction='forward').fit(self.data, self.labels)
        # Return the selected features
        return self.data.loc[:, sfs.get_support()].columns

#%% 1- Pre Processing and EDA

path = 'data/heart_2020.csv'
data_analysis_instance = DataAnalysis(path, 'HeartDisease')

data_analysis_instance.describe_variables()

# Process all the data to numeric values
data_analysis_instance.process_data()

# Determine the range of values for each variable
data_analysis_instance.determine_range()

# Verify the presence of duplicated data and remove it
data_analysis_instance.assess_quality()

# Plots after the cleansing and processing
#data_analysis_instance.plots(['correlation'])

# Cleaned CSV
path_cleaned = 'data/heart_2020_cleaned.csv'

#pca_analysis = PCAAnalysis(path_cleaned, 2)

#pca_analysis.display_feature_contributions()
#pca_analysis.calculate_explained_variance_ratio()
#pca_analysis.plot_explained_variance_ratio()
#pca_analysis.print_pca_projection()
#pca_analysis.plot_pca_projections()

# Initialize DimensionalityReduction object with the dataset
dr = DimensionalityReduction(path_cleaned)

# Compute and plot PCA projection
# dr.plot_projection(dr.compute_pca(), 'PCA Projection')
# Compute and plot LDA projection
# dr.plot_projection(dr.compute_lda(), 'LDA Projection')
# Compute and plot t-SNE projection
# dr.plot_projection(dr.compute_tsne(), 't-SNE Projection')
# Compute and plot UMAP projection
# dr.plot_projection(dr.compute_umap(), 'UMAP Projection')
# Compute and plot LLE projection
# dr.plot_projection(dr.compute_lle(), 'LLE Projection')

# Initialize FeatureExtractor object with the dataset
extractor = FeatureExtractor(path_cleaned)

# Extract features
feature_df = extractor.extract_features()
# Display the dataframe
print(feature_df)
# Plot the features per class
extractor.plot_all_features()

#%% 2- Hypothesis Testing

#%% 3- Modeling
