
from typing import Dict, List

#from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
from environment import channels_good

from train_models import split_generator

def caculate_mode_all(model: SVC, X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list) -> List:
    """ Calculate accuracy for each channel (Acc_i) by training on the whole dataset """

    channel_acc: Dict[str, float] = {}

    for ch in tqdm(channels_good):
        X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]

        y_train = y_train_org["is_fatigued"]
        y_test = y_test_org["is_fatigued"]

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

            y_train = y_train_org["is_fatigued"]
            y_test = y_test_org["is_fatigued"]

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
    training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    X = df.drop("is_fatigued", axis=1)
    y = df.loc[:, "is_fatigued"]
    X_train, X_test, y_train, y_test= split_generator(X, y)
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    caculate_mode_all(model, X_train, X_test, y_train, y_test, channels_good)