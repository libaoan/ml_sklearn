# -*- coding:utf-8 -*-

from numpy import *
import jieba

def load_data_set():
    """
    创建数据集
    :return: data_set, target
    """
    posting_list = [
        "你他娘的，明天这个PPT交上来不，你他娘的就滚蛋， 妈的...",
        "二营长，在家吗？ 赵刚他娘的今天出差了，今晚你就滚回到老子家来吧",
        "滚吧，滚得越远越好，老子再也不想看到你",
        "二营长，你他娘的死哪去了",
        "我估摸着城门楼子是快难啃的骨头，老子就是崩了门牙，也要在鬼子的增援部队赶到前咬开它！",
        "都说鬼子拼刺刀有两下子，老子就不信这个邪，都是两个肩膀扛一个脑袋，鬼子他是人养的，肉长的，\
            大刀进去也要穿个窟窿。就算是见了阎王爷，老子也能撸它几根胡子下来！",
        "什么他娘的精锐，老子打的就是精锐。",
        "当你成功的时候，你说的所有话都是真理。",
        "狭路相逢,勇者胜",
        "没有助攻，全他娘的主攻、现在我们的兵力是八比一，这种富裕仗我八辈子也没打过，这会咱们敞开了当回地主。 \
            三营长，你嘴别裂的跟荷花式的，助攻改主攻，我一不给添人，二不给添枪，一字之变，要给我变出杀气来，要打出个精神头来",
        "我自己搞武器，行啊，你不能限制我的自由啊，总要点自主权吧！又要我当乖孩子，又要我自己想办法搞武器，\
            又限制我的自主权，这叫不讲道理",
        "我永远相信只要永不放弃，我们还是有机会的。最后，我们还是坚信一点，这世界上只要有梦想，只要不断努力，\
            只要不断学习，不管你长得如何，不管是这样，还是那样，男人的长相往往和他的的才华成反比。",
        "孙正义跟我有同一个观点，一个方案是一流的Idea加三流的实施；另外一个方案，一流的实施加三流的Idea，哪个好？\
            我们俩同时选择一流的实施，三流的Idea。"
        "什么他娘的武士道,老子打的就是武士道",
        ]
    target = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]

    return posting_list, target


def cut_words(text):
    """
    中文截词， 返回经过截词的字符串
    :param text:
    :return:
    """
    res = jieba.cut(text)
    return list(res)


def data_set_extraction(data_set):
    """
    获取所有单词的集合
    :param data_set: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    res = set([])  # create empty set
    for data in data_set:
        # 操作符 | 用于求两个集合的并集
        res = res | set(data)  # union of the two sets
    return list(res)


def get_counted_words(words_list, post):

    res = [0] * len(words_list)
    for word in post:
        if word in words_list:
            res[words_list.index(word)] = 1
    return res



def trainNB(train_matrix, targets):
    """
    训练数据优化版本
    :param train_matrix: 文件单词矩阵
    :param targets: 文件对应的类别
    :return:
    """
    # 总文件数
    cnt_post = len(train_matrix)
    # 总单词数
    cnt_words = len(train_matrix[0])
    # 侮辱性文件的出现概率
    pt_v = sum(targets) / float(cnt_post)
    # 构造单词出现次数列表
    # p0 正常的统计
    # p1 侮辱的统计
    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    p0_num = ones(cnt_words)  #[0,0......] -> [1,1.....]
    p1_num = ones(cnt_words)

    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0 正常的统计
    # p1 侮辱的统计
    p0 = 2.0
    p1 = 2.0
    for i in range(cnt_post):
        if targets[i] == 1:
            # 累加辱骂词的频次
            p1_num += train_matrix[i]
            # 对每篇文章的辱骂的频次 进行统计汇总
            p1 += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0 += sum(train_matrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1_v = log(p1_num / p1)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0_v = log(p0_num / p0)
    return p0_v, p1_v, pt_v


def classify(vec2Classify, p0_v, p1_v, pt_v):
    """
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    # 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
    p1 = sum(vec2Classify * p1_v) + log(pt_v)
    p0 = sum(vec2Classify * p0_v) + log(1.0 - pt_v)
    if p1 > p0:
        return "李云龙"
    else:
        return "马云"


def testing():
    """
    测试朴素贝叶斯算法
    """
    # 1. 加载数据集，并分词转换为列表格式
    posts, target_list = load_data_set()
    post_list = []
    for post in posts:
        post_list.append(cut_words(post))
    # 2. 创建单词集合
    words_list = data_set_extraction(post_list)
    # 3. 计算单词是否出现并创建数据矩阵
    train_matrix = []
    for post in post_list:
        # 返回m*len(words_list)的矩阵， 记录的都是0，1信息
        train_matrix.append(get_counted_words(words_list, post))
    # 4. 训练数据
    p0_v, p1_v, pt_v = trainNB(array(train_matrix), array(target_list))
    # 5. 测试数据
    post = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"
    test_post = cut_words(post)
    thisDoc = array(get_counted_words(words_list, test_post))
    print(post, 'classified as: ', classify(thisDoc, p0_v, p1_v, pt_v))
    post = "二营长,你他娘的意大利炮呢?给老子拉上来!"
    test_post = cut_words(post)
    thisDoc = array(get_counted_words(words_list, test_post))
    print(post, 'classified as: ', classify(thisDoc, p0_v, p1_v, pt_v))


if __name__ == "__main__":
    testing()