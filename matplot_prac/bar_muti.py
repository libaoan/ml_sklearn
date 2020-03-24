# -*- encoding:utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import font_manager

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

a = ["战狼2", "速度与激情8", "功夫瑜伽", "西游伏魔牌", "变形金刚5"]
day_15 = [56.01, 26.94, 17.53, 16.49, 15.45]
day_16 = [78.23, 34.32, 22.32, 19.32, 22.32]
day_17 = [99.21, 33.33, 32.35, 23.45, 23.24]

width = 0.2
x_15 = range(len(a))
x_16 = [i + width for i in x_15]
x_17 =[i + 2*width for i in x_15]
plt.figure(figsize=(20, 8), dpi=80)

plt.bar(x_15, day_15, width=width, label="15日")
plt.bar(x_16, day_16, width=width, label="16日")
plt.bar(x_17, day_17, width=width, label="17日")

# 设置X的刻度
plt.xticks(x_16, a, fontproperties=font, rotation=45)

plt.grid(alpha=0.3)

# 添加图例
plt.title("3月份电影票房统计", fontproperties=font)
plt.legend(prop=font)
plt.show()
