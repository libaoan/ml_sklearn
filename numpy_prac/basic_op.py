# -*- encoding:utf-8 -*-

import numpy as np

t1 = np.arange(12)
print(t1)
print(t1.shape)

t2 = np.array([[1, 2, 3], [4, 5, 6]])
print(t2)
print(t2.shape)

t3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]],])
print(t3)
print(t3.shape)

t4 = t1
print(t4.reshape((3, 4)))

t5 = np.arange(24)
print(t5)
t5 = t5.reshape(4, 6)
print(t5.reshape((4, 6)))
print(t5.reshape((24, )))
print(t5.reshape((24, 1)))
print(t5.reshape((1, 24)))

t6 = t5.reshape((t5.shape[0]*t5.shape[1]), )
print(t6)
# print(t5.flatten())

# 运算
print(t5)
t = t5 + 2
print(t)
print("-"*80)
t = t5 / 0
print(t)
print("-"*80)

t6 = np.arange(0, 6)
t = t5 + t6
print(t)
print("-"*80)

t6 = np.arange(0, 4).reshape((4, 1))
t = t5 * t6
print(t)
print("-"*80)

t6 = np.arange(100, 124).reshape(4, 6)
t = t5 + t6
print(t)
print("-"*80)

t5 = t5.reshape((2, 3, 4))
t7 = np.arange(100, 112).reshape((3, 4))
t = t5 + t7
print(t)
print("-"*80)

