# -- encoding: utf-8 --
from sklearn.feature_selection import VarianceThreshold


def variance_select():
    """
    特征选择-删除底方差的特性
    :return: None
    """
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


if __name__ == "__main__":
    variance_select()