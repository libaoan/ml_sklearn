# -- encoding: utf-8 --

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter  # 为了做投票


# raw_data_x属于特征值，分别属于患病时间和肿瘤大小
raw_data_x = [[3.3935, 2.3312],
              [3.1101, 1.7815],
              [1.3438, 3.3684],
              [3.5823, 4.6792],
              [2.2804, 2.8670],
              [7.4234, 4.6965],
              [5.7451, 3.5340],
              [9.1722, 2.5111],
              [7.7928, 3.4241],
              [7.9398, 0.7916]]

# raw_data_y属于目标值，0代表良性肿瘤，1代表恶性肿瘤
row_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def show_plot(x_train, y_train):
    """
    绘图
    :param x_train:
    :param y_train:
    :return:
    """
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color="g")
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color="r")
    plt.xlabel('Tumor Size')
    plt.ylabel("Time")
    plt.axis([0, 10, 0, 5])
    plt.show()


class KNNClassifier:

    def __init__(self, k):
        assert k > 1, "K值必须大于1"
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train和y_train训练K分类器
        x_train: 训练数据的特征
        y_train: 训练数据的标签
        """
        assert x_train.shape[0] == y_train.shape[0], "x_train的大小必须等于y_train的大小"
        assert self.k <= x_train.shape[0], "x_train的大小必须大于K"
        self.x_train = x_train
        self.y_train = y_train

    def euc_dis(self, instance1, instance2):
        """
        计算两个样本instance1和instance2之间的欧式距离
        instance1: 第一个样本， array型
        instance2: 第二个样本， array型
        """
        dist = np.sqrt(sum((instance1 - instance2) ** 2))
        return dist

    def knn_classify(self, testInstance):
        """
        给定一个测试数据testInstance, 通过KNN算法来预测它的标签。
        testInstance: 测试数据，这里假定一个测试数据 array型
        """
        distances = [self.euc_dis(x, testInstance) for x in self.x_train]
        kneighbors = np.argsort(distances)[:self.k]
        count = Counter(self.y_train[kneighbors])
        return count.most_common()[0][0]

    def score(self, x_test, y_test):
        """根据测试样本x_test进行预测，和真实的目标值进行比较，计算预测结果的准确度"""

        predictions = [self.knn_classify(x) for x in x_test]
        correct = np.count_nonzero((predictions == y_test) == True)
        return float(correct)/len(x_test)

    def __repr__(self):
        return "KNN(k={})".format(self.k)


if __name__ == "__main__":
    # 从数据集中选择2个样本为测试样本, 其余的均为训练样本
    x_train = np.array(raw_data_x[:-2])
    y_train = np.array(row_data_y[:-2])

    x_test = np.array(raw_data_x[-2:])
    y_test = np.array(row_data_y[-2:])

    # 原始数据绘图
    # show_plot(np.array(raw_data_x), np.array(row_data_y))

    knn = KNNClassifier(3)
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print("测试精确度为: %.3f" % score)

    # 未知的待分类样本
    x_test2 = np.array([8.9093, 3.3657])
    result = knn.knn_classify(x_test2)
    if result == 0: result = "良性肿瘤"
    else: result = "恶性肿瘤"
    print("对[8.9093, 3.3657]的预测结果为:【%s】" % result)
