"""
k-近邻算法概述
简单地说，k近邻算法采用测量不同特征值之间的距离方法进行分类。

k-近邻算法

优点：精度高、对异常值不敏感、无数据输入假定。
缺点：计算复杂度高、空间复杂度高。 适用数据范围：数值型和标称型。
      k-近邻算法（kNN)，它的工作原理是：存在一个样本数据集合，也称作训练样本集，
并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输
入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法
提取样本集中特征最相似数据（最近邻）的分类标签。一般来说，我们只选择样本数据集中前
k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。最后，选择k个最
相似数据中出现次数最多的分类，作为新数据的分类。

k近邻算法的一般流程

收集数据：可以使用任何方法。
准备数据：距离计算所需要的数值，最好是结构化的数据格式。
分析数据：可以使用任何方法。
训练算法：此步骤不适用于k近邻算法。
测试算法：计算错误率。
使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。
"""

# k-neighbour算法  需要做标准化处理

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets   #引用数据集
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

def knncls():
    #读取数据
    iris = datasets.load_iris()
    X = iris.data
    print('X:\n',X)
    Y = iris.target
    print('Y:\n',Y)

    #处理二分类问题，所以只针对Y= 0,1 的行，然后从这些行中取X的前两列
    x = X[Y<2,:2]
    print(x.shape)
    print('x:\n',x)
    y = Y[Y<2]
    print('y:\n',y)


    #target=0的点标红，target=1的点标蓝,点的横坐标为data的第一列，点的纵坐标为data的第二列
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color='green')
    plt.scatter(5.6, 3.2, color='blue')
    x_1 = np.array([5.6, 3.2])
    plt.title('红色点标签为0,绿色点标签为1，待预测的点为蓝色')
    plt.show()
    # 采用欧式距离计算
    distances = [np.sqrt(np.sum((x_t - x_1) ** 2)) for x_t in x]
    # 对数组进行排序，返回的是排序后的索引
    d = np.sort(distances)
    nearest = np.argsort(distances)
    k = 6
    topk_y = [y[i] for i in nearest[:k]]
    from collections import Counter
    # 对topk_y进行统计返回字典
    votes = Counter(topk_y)
    # 返回票数最多的1类元素
    print(votes)
    predict_y = votes.most_common(1)[0][0]
    print(predict_y)
    plt.show()
    return None

if __name__ == '__main__':
    knncls()