# -*- encoding:utf-8 -*-

from matplotlib import font_manager
from matplotlib import pyplot as plt
import random

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

y = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
x = range(11, 31)

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y)

x_labels = ["{}岁".format(i) for i in range(11, 31)]
plt.xticks(x, x_labels, fontproperties=font)
plt.yticks([i for i in range(0, 10)])

# 绘制网格
plt.grid(alpha=0.2)

plt.show()