
from typing import Dict, List

#from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (train_test_split)
from sklearn.svm import SVC
from tqdm import tqdm
from environment import channels_good
from pandas import DataFrame, read_pickle
#from train_models import split_generator
from helper_functions import (glimpse_df,isnull_any, rows_with_null)
from preprocess import (df_replace_values)
#from models import model_svc


def caculate_mode_all(model: SVC, X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list) -> List:
    """ Calculate accuracy for each channel (Acc_i) by training on the whole dataset """

    channel_acc: Dict[str, float] = {}

    for ch in tqdm(channels_good):
        X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]
        #y_train = y_train_org["is_fatigued"]
        y_train = y_train_org.values
        #y_test = y_test_org["is_fatigued"]
        y_test = y_test_org.values

        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        channel_acc[ch] = accuracy_score(y_test, y_test_pred)

    """ Calculate weight for the whole dataset for each channel (V_i). """

    channel_weights = {}
    for channel_a_name in tqdm(channels_good):
        sum_elements = []
        for channel_b_name in channels_good:
            """ Calculate Acc(i,j) and add it to sum expression """
            if channel_b_name == channel_a_name:
                break

            X_train = X_train_org.loc[:, X_train_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

            X_test = X_test_org.loc[:, X_test_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

            # y_train = y_train_org["is_fatigued"]
            y_train = y_train_org.values
            # y_test = y_test_org["is_fatigued"]
            y_test = y_test_org.values

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            acc_ij = accuracy_score(y_test, y_test_pred)
            print(acc_ij)
            sum_elements.append(acc_ij + channel_acc[channel_a_name] - channel_acc[channel_b_name])

        sum_expression = sum(sum_elements)
        acc_i = channel_acc[channel_a_name]
        weight = (acc_i + sum_expression) / len(channels_good)
        channel_weights[channel_a_name] = weight

    return sorted(channel_weights.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
    df: DataFrame = read_pickle(df_path)
    glimpse_df(df)
    channels_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ",
                     "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]



    # Load data and labels
    X = df.drop("is_fatigued", axis=1)
    print(X)
    y = df.loc[:, "is_fatigued"]
    print(y)
    #print(y["is_fatigued"])
    # Split data into training and testing sets
    #X_train, X_test, y_train, y_test = split_generator(X, y)
    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.5, shuffle=True)
    model =SVC(kernel="rbf")
    res=caculate_mode_all(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good)
    print(res)