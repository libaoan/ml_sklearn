# -- encoding: utf-8 --

import numpy as np
import matplotlib.pyplot as plt

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

X_train = np.array(raw_data_x)
y_train = np.array(row_data_y)

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="g")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="r")
plt.ylabel('Tumor Size')
plt.xlabel("Time")
plt.axis([0, 10, 0, 5])
plt.show()
