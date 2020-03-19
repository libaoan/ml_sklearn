# -*- encoding:utf-8 -*-

import numpy as np

# 数据集包含点的个数
m = 20

# Points x-coordinate and dummy value (x0, x1)
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))
# print(X0, X1, X)

# Points y-coordinate
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
# print(y)

# 超参数 步长 alpha
alpha = 0.01


def error_function(theta, X, y):
    """Error function J definition"""
    diff = np.dot(X, theta) - y
    return (1.0/2*m) * np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    """Gradient of the function J definition"""
    diff = np.dot(X, theta) - y
    return (1.0/m) * np.dot(np.transpose(X), diff)


def gradient_descent(X, y, alpha):
    """Perform gradient descent"""
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta


optimal = gradient_descent(X, y, alpha)
print("最小值:\n", optimal)
print("损失函数:\n", error_function(optimal, X, y)[0, 0])