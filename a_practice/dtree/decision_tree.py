#!/usr/bin/python
# coding:utf-8

import operator
from math import log
import matplotlib.pyplot as plt


def create_data_set():
    """data_set 基础数据集
    args:
        无需传入参数
    returns:
        返回数据集和对应的label标签
    """
    # 特征分别为：长相、学历、幽默、收入、是否喜欢
    data_set = [[0, 1, 0, 1, 'yes'],
               [1, 1, 1, 1, 'yes'],
               [0, 0, 1, 0, 'no'],
               [1, 0, 1, 1, 'no'],
               [1, 1, 0, 1, 'yes'],
               [0, 0, 1, 0, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [1, 0, 0, 1, 'no'],
                ]
    labels = ['is_beautiful', 'is_educated', 'is_funny', 'is_rich']
    # change to discrete values
    return data_set, labels


def calc_shannon_ent(data_set):
    """calc_shannon_ent(calculate Shannon entropy 计算给定数据集的香农熵)
    Args:
        data_set 数据集
    Returns:
        返回 每一组feature下的某个分类下，香农熵的信息期望
    """
    # 求list的长度，表示计算参与训练的数据量
    num_entries = len(data_set)
    # 计算分类标签label出现的次数
    labels = {}
    # 计算目标熵
    for instance in data_set:
        target = instance[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if target not in labels.keys():
            labels[target] = 0
        labels[target] += 1

    # 对于label标签的占比，求出label标签的香农熵
    shannon_ent = 0.0
    for key in labels:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labels[key]) / num_entries
        # log base 2
        # 计算香农熵，以 2 为底求对数
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, index, value):
    """shannon_ent(通过遍历data_set数据集，求出index对应特征列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        data_set 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    ret_data_set = []
    for instance in data_set:
        if instance[index] == value:
            target_instance = instance[:index]
            target_instance.extend(instance[index + 1:])
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】
            ret_data_set.append(target_instance)
    return ret_data_set


def select_best_feature(data_set):
    """select_best_feature(选择最高增益的特征)
    Args:
        data_set 数据集
    Returns:
        best_feature 最优的特征列
    """

    # 求第一行有多少列的 Feature, 最后一列是label列嘛
    num_features = len(data_set[0]) - 1
    # label的信息熵
    base_entropy = calc_shannon_ent(data_set)
    # 最优的信息增益值, 和最优的featurn编号
    best_gain, best_feature = 0.0, -1
    # iterate over all the features
    for i in range(num_features):
        # 获取每一个实例的同一个feature，组成list集合
        feature_list = [instance[i] for instance in data_set]
        # 获取剔重后的集合，使用set对list数据进行去重
        unique_vals = set(feature_list)
        # 创建一个临时的信息熵
        new_entropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        info_gain = base_entropy - new_entropy
        print('infoGain=', info_gain, 'bestFeature=', i, base_entropy, new_entropy)
        if info_gain >= best_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """majority_cnt(选择出现次数最多的一个结果)
    Args:
        class_list
    Returns:
        best_future 最优的特征列
    """
    class_cnt = {}
    for vote in class_list:
        if vote not in class_cnt.keys():
            class_cnt[vote] = 0
        class_cnt[vote] += 1
    # 倒叙排列class_cnt得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sorted_class_cnt = sorted(class_cnt, key=operator.itemgetter(1), reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sorted_class_cnt[0][0]


def make_tree(data_set, labels):
    class_list = [instance[-1] for instance in data_set]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    # 选择最优的列，得到最优列对应的label含义
    best_feature = select_best_feature(data_set)
    # 获取label的名称
    best_feature_label = labels[best_feature]
    # 初始化myTree
    my_tree = {best_feature_label: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[best_feature])
    # 取出最优列，然后它的branch做分类
    feature_values = [instance[best_feature] for instance in data_set]
    unique_vals = set(feature_values)
    for value in unique_vals:
        # 求出剩余的标签label
        sub_labels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数make_tree()
        sub_data_set = split_data_set(data_set, best_feature, value)
        my_tree[best_feature_label][value] = make_tree(sub_data_set, sub_labels)

    return my_tree


def classify(input_tree, featrue_labels, test_value):
    """classify(给输入的节点，进行分类)
    Args:
        input_tree  决策树模型
        featrue_labels Feature标签对应的名称
        test_value    测试输入的数据
    Returns:
        class_label 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    first_str = list(input_tree.keys())[0]
    # 通过key得到根节点对应的value
    second_dict = input_tree[first_str]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featrue_index = featrue_labels.index(first_str)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = test_value[featrue_index]
    value_of_feature = second_dict[key]
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(featrue_index, dict):
        class_label = classify(value_of_feature, featrue_labels, test_value)
    else:
        class_label = featrue_labels
    return class_label


def store_tree(input_tree, filename):
    import pickle
    # -------------- 第一种方法 start --------------
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# 定义文本框 和 箭头格式 【 sawtooth 波浪方框, round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅，没错是变浅】
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 记录最大的分支深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)
    # 获取树的深度
    # depth = getTreeDepth(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    # x坐标为 (numLeafs-1.)/plotTree.totalW/2+1./plotTree.totalW，化简如下
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # print cntrPt
    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]
    # 可视化Node分支点；第一次调用plotTree时，cntrPt与parentPt相同
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 根节点的值
    secondDict = myTree[firstStr]
    # y值 = 最高点-层数的高度[第二个节点位置]；1.0相当于树的高度
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    # 创建一个figure的模版
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 半个节点的长度；xOff表示当前plotTree未遍历到的最左的叶节点的左边一个叶节点的x坐标
    # 所有叶节点中，最左的叶节点的x坐标是0.5/plotTree.totalW（因为totalW个叶节点在x轴方向是平均分布在[0, 1]区间上的）
    # 因此，xOff的初始值应该是 0.5/plotTree.totalW-相邻两个叶节点的x轴方向距离
    plotTree.xOff = -0.5 / plotTree.totalW
    # 根节点的y坐标为1.0，树的最低点y坐标为0
    plotTree.yOff = 1.0
    # 第二个参数是根节点的坐标
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()



def test():
    # 1.创建数据和结果标签
    data, labels = create_data_set()


    import copy
    my_tree = make_tree(data, copy.deepcopy(labels))
    print(my_tree)
    # [0, 1, 0, 0]为待测样本，代表（长相一般、高学历、无趣味、没钱）
    print(classify(my_tree, labels, [0, 1, 0, 0]))

    # 获得树的高度
    print(get_tree_height(my_tree))

    # 画图可视化展现
    createPlot(my_tree)


def get_tree_height(tree):
    """
     Desc:
        递归获得决策树的高度
    Args:
        tree
    Returns:
        树高
    """

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1


if __name__ == "__main__":
    test()