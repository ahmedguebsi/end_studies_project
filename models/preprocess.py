import warnings
import numpy as np
from pandas import DataFrame, Series, read_pickle, set_option
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import MinMaxScaler

from environment import training_columns_regex

def df_filter_columns_by_std(X_train: Series, X_test: Series, std=0.01):
    X_test = X_test.loc[:, X_train.std() > 0.1]
    X_train = X_train.loc[:, X_train.std() > 0.1]
    return X_train, X_test


def split_and_normalize(X: Series, y: Series, test_size: float, columns_to_scale, scaler: MinMaxScaler = MinMaxScaler()):
    """Columns to scale can be both string list or list of bools"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    X_train: Series
    X_test: Series
    y_train: Series
    y_test: Series

    X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train.loc[:, columns_to_scale])
    X_test.loc[:, columns_to_scale] = scaler.transform(X_test.loc[:, columns_to_scale])
    return X_train, X_test, y_train, y_test


def df_replace_values(df: DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    set_option("display.max_columns", None)
    warnings.filterwarnings("ignore")


    df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-raw-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
    output_dir= r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
    df: DataFrame = read_pickle(df_path)
    print(df.head())
    training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    df = df_replace_values(df)
    print(df.head())

    basename = df_path.stem.replace("raw", "cleaned2")
    file_saver = lambda df, filepath: DataFrame.to_pickle(df, filepath)
    filepath = get_decorated_filepath(directory=output_dir, basename=basename, extension=".pkl")
    save_obj(obj=df, filepath=filepath, file_saver=DataFrame.to_pickle, metadata={})