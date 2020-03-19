# -*- encoding:utf-8 -*-

from matplotlib import font_manager
from matplotlib import pyplot as plt
import random

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

y1 = [1,0,1,1,2,3,3,2,3,4,9,8,6,4,4,2,3,2,1,1]
y2 = [1,0,3,4,5,5,3,2,2,3,2,1,3,2,2,1,1,1,1,1]
x = range(11, 31)

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y1, color="r", label="小明交往女友的数量", linewidth=3, linestyle=":", alpha=0.5)
plt.plot(x, y2, color="#DB7893", label="小王交往的女友数量", linewidth=2, linestyle="-.", alpha=0.9)

x_labels = ["{}岁".format(i) for i in range(11, 31)]
plt.xticks(x, x_labels, fontproperties=font)
plt.yticks([i for i in range(0, 10)])

# 绘制网格
plt.grid(alpha=0.2, linestyle="--")

# 添加图例
plt.legend(prop=font, loc="best")

plt.show()