# -- encoding: utf-8 --

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np


def min_max():

    """
    归一化处理
    :return: None
    """
    prep = MinMaxScaler(feature_range=(-1.2, 1.5))
    # prep = MinMaxScaler()
    data = prep.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [45, 1, 5, 20]])
    print(data)
    return None


def standard():
    """
    标准化处理
    :return: None
    """
    prep = StandardScaler()
    data = prep.fit_transform([[90, 2222, 10, 40], [60, 4000, 15, 45], [45, 1222, 5, 20]])
    print(data)
    return None


def inputer():
    """
    缺失值填补
    :return: None
    """
    ins = SimpleImputer(missing_values="NaN", strategy="mean")
    data = ins.fit_transform([[2, np.nan, 3], [2, 4, np.nan]])
    print(data)
    return None


if __name__ == "__main__":
    # min_max()
    # standard()
    inputer()