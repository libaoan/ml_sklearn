# -*- encoding:utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib import font_manager

font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\msyh.ttf")

a = ["战狼2", "速度与激情8", "功夫瑜伽", "西游伏魔牌", "变形金刚5", "天线宝宝之\n大战三百回合"]
b = [56.01, 26.94, 17.53, 16.49, 15.45, 11.2]


plt.figure(figsize=(20, 8), dpi=80)

# plt.bar(range(len(a)), b, width=0.3)
# plt.xticks(range(len(a)), a, fontproperties=font, rotation=45)

# 横着放置直方图
plt.barh(range(len(a)), b, height=0.3)
plt.yticks(range(len(a)), a, fontproperties=font, rotation=45)


# 添加图例

plt.title("3月份电影票房统计", fontproperties=font)
plt.show()
