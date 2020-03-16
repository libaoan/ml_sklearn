# -*- encoding:utf-8 -*-


import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plt
import random

# Windows和linux设置字体方式
# font = {'family': 'Microsoft YaHei',
#         'weight': 'bold',
#         'size': '8'}
# matplotlib.rc('font', **font)
# matplotlib.rc('font', family="Microsoft YaHei", size=8)

# 另一种方法
font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")


x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y)

# 调整X轴的刻度
x = list(x)
x_ticks = ["2点{}分".format(i) for i in range(60)]
x_ticks += ["3点{}分".format(i) for i in range(60)]
# 刻度与标签对应, #rotation标签旋转
plt.xticks(x[::3], x_ticks[::3], rotation=45, fontproperties=font)

plt.show()