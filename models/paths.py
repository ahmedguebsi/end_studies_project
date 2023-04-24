
from os import getcwd
from pathlib import Path

PATH_CWD = Path(getcwd())
print(PATH_CWD)
PATH_DATA = Path(PATH_CWD, "Data")
PATH_DATASET = Path(PATH_DATA, "dataset.ignoreme")
print(PATH_DATASET)
PATH_DATAFRAME = Path(PATH_DATA, "dataframes")
PATH_DATASET_MAT = Path(PATH_DATASET, "mat")
PATH_DATASET_CNT = Path(PATH_DATASET, "cnt")
print(PATH_DATASET_CNT)
PATH_ZIP_CNT = Path(PATH_DATASET, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATASET, "5202751.zip")
PATH_MODEL = Path(PATH_CWD, "models")
PATH_REPORT = Path(PATH_CWD, "reports")
PATH_FIGURE = Path(PATH_REPORT, "figures")

if __name__ == "__main__":
    pass