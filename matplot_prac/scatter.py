# -*- encoding:utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import font_manager

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

y3 = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22, 22,
      23, 22]
y10 = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23, 17, 20, 21, 20, 22, 15, 11, 15, 5, 13, 17, 10, 11, 13,
       12, 11, 9]

x3 = range(1, 32)
x10 = range(51, 82)

plt.figure(figsize=(20, 8), dpi=80)

plt.scatter(x3, y3, label="3月")
plt.scatter(x10, y10, label="10月")

_x = list(x3) + list(x10)
x_tick_label = ["3月{}日".format(i) for i in x3]
x_tick_label += ["10月{}日".format(i-50) for i in x10]

plt.xticks(_x[::3], x_tick_label[::3], fontproperties=font, rotation=45)

# 添加图例
plt.legend(loc="upper left", prop=font)


# 添加图片描述
plt.xlabel("时间", fontproperties=font)
plt.ylabel("温度", fontproperties=font)
plt.title("北京地区3月及10月温度散点图", fontproperties=font)
plt.show()
