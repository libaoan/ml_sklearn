# -*- encoding:utf-8 -*-

import numpy as np
import random

t1 = np.array([i for i in range(12)])
t2 = np.arange(0, 12)
print(t1, t1.dtype)
print(t2, t2.dtype)

t3 = np.array(range(1, 4), dtype=float)
print(t3.dtype)
t3 = np.array(range(1, 4), dtype="float32")
print(t3.dtype)

t4 = np.array([1, 0, 1, 1, 0], dtype=bool)
print(t4, t4.dtype)
t5 = t4.astype("int8")
print(t5, t5.dtype)

t6 = np.array([random.random() for i in range(6)])
print(t6, t6.dtype)
t6 = np.round(t6, 2)
print(t6, t6.dtype)
