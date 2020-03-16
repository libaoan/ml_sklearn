
from matplotlib import pyplot as plt


fig = plt.figure(figsize=(10, 8), dpi=80)

x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 27, 26, 24, 22, 18, 15]


plt.plot(x, y)

# 设置x.y轴的刻度
# x_ticks = [i/2 for i in range(4, 50)]
# 设置为每隔1小时一个刻度
x_ticks = [i for i in range(2, 24)]
plt.xticks(x_ticks[::])
y_ticks = [i for i in range(min(y), max(y)+1)]
plt.yticks(y_ticks)

plt.savefig("./figure.png")
plt.show()
