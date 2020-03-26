# -*- encoding:utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import font_manager

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]
width = [5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 30, 60]
quantity = [836, 2737, 3723, 3926, 3596, 1438, 3273, 642, 824, 613, 215, 47]

plt.figure(figsize=(20, 8), dpi=80)

plt.bar(range(len(interval)), quantity, width=1)


_x = [i for i in range(len(interval)+1)]
_x_labels = interval.append(150)
plt.xticks(_x, _x_labels)

# 添加图例
plt.title("伪直方统计图", fontproperties=font)
plt.grid(alpha=0.3)
plt.show()
