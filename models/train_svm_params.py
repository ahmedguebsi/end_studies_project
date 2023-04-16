"""
Finds the best C and gamma hyperparameters for SVM model by using Leave One Group out approach.

A single group of rows is defined by participant's id (driver_id).
Effectively, this is LOO approach where 1 participant is left for validation and other 11 are used for training the model

Load the dataset with the --df argument
Calculate the accuracy for each hyperparameter pair
"""
import argparse
from itertools import product
from pathlib import Path

from pandas import DataFrame, read_pickle
from pandas._config.config import set_option
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from tqdm import tqdm

from models import wide_params
from preprocess import split_and_normalize, df_replace_values
from environment import NUM_USERS, training_columns_regex
from helper_functions import get_timestamp, glimpse_df, stdout_to_file



#df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-raw-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
#stdout_to_file(Path(args.output_report, "-".join(["svm-parameters", get_timestamp()]) + ".txt"))

df: DataFrame = read_pickle(df_path)
print(df.head())
training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
print(training_columns)
#df_replace_values(df)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
X = X[X.columns[X.max() != -1]]  # remove constant attributes
y = df.loc[:, "is_fatigued"]

groups = X["driver_id"].to_numpy()
acc_parameters = []

for C, gamma in tqdm(list(product(wide_params, wide_params))):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    acc_total = 0

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        acc_total += acc
    acc_parameters.append([acc_total / NUM_USERS, C, gamma])

print("Acc\t\t\tC\tgamma")
accs = sorted(acc_parameters, key=lambda x: x[0], reverse=True)
for acc in accs:
    print(acc)