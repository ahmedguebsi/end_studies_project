import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import fisher_score
from skfeature.function.sparse_learning_based import ls_l21

from skfeature.utility.construct_W import construct_W
#from sklearn.neighbors import construct_W
#from skfeature.function.similarity_based import construct_W

from scipy.linalg import eigh

from environment import channels_good
from pandas import DataFrame, read_pickle
from sklearn.model_selection import (train_test_split)
from helper_functions import (get_cnt_filename, glimpse_df, serialize_functions,isnull_any)
from preprocess import (df_replace_values, rows_with_null)


class HSFCS(BaseEstimator, TransformerMixin):

    def __init__(self, n_channels=30, n_features=100, k=5, alpha=0.1, beta=0.1):
        self.n_channels = n_channels  # number of channels to select
        self.n_features = n_features  # number of features to select
        self.k = k  # number of nearest neighbors for mutual information calculation
        self.alpha = alpha  # weight for channel selection
        self.beta = beta  # weight for feature selection

    def fit(self, X, y):
        # Calculate mutual information between each channel and the target variable
        mi = mutual_info_classif(X, y, n_neighbors=self.k)

        # Select the most informative channels
        selected_channels = np.argsort(mi)[-self.n_channels:]

        # Calculate similarity matrix based on feature correlation
        W = construct_W(X.T, k=5)

        # Calculate fisher score for each feature
        fisher_score = fisher_score.fisher_score(X[:, selected_channels], y, W=W)

        # Select the most informative features
        feature_idx = ls_l21.ls_l21(X[:, selected_channels], y, self.n_features, self.alpha, self.beta)
        selected_features = selected_channels[feature_idx]

        self.selected_channels = selected_channels
        self.selected_features = selected_features

        return self

    def transform(self, X):
        # Select the most informative channels and features
        X = X[:, self.selected_channels]
        X = X[:, self.selected_features]

        return X

if __name__ == "__main__":
    df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
    df: DataFrame = read_pickle(df_path)
    glimpse_df(df)
    print(isnull_any(df))
    print(rows_with_null(df))
    df = df_replace_values(df)
    print(isnull_any(df))
    print(rows_with_null(df))
    channels_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ",
                         "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
    training_columns_regex = "|".join(channels_good)
    training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    glimpse_df(df)
    # Load data and labels
    X = df.drop("is_fatigued", axis=1)
    y = df.loc[:, "is_fatigued"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test= train_test_split(X, y)
    print(X_train)
    # Create HSFCS instance and fit to training data
    hsfcs = HSFCS(n_channels=30, n_features=100, k=5, alpha=0.1, beta=0.1)
    print(type(hsfcs))
    hsfcs.fit(X_train, y_train)
    #hsfcs.fit(X, y)

    # Transform training and testing data
    X_train = hsfcs.transform(X_train)
    X_test = hsfcs.transform(X_test)

    # Train KNN classifier on transformed data and evaluate performance
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train.values)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
