from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer   #特征化处理
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np
# 特征抽取
#
# 导入包

# feature_extraction   特征 抽取   CountVectorizer  计数并返回向量化程序
# from sklearn.feature_extraction.text import CountVectorizer

#实例化CountVectorizer
# vector = CountVectorizer()

# 调用fit——transform输入并转换数据

# 特征抽取对文本等数据进行特征值化，是为了计算机更好的去理解数据
# res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])

# 打印结果
# 返回类别名称
# print(vector.get_feature_names())

# print(res.toarray)


def dictvec():

    # 对字典数据进行处理   数值化使计算机可以理解

    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])
    # 打印输出类别名称
    print(dict.get_feature_names())
    # 将得到的数字化数据再转换为字典数据
    print(dict.inverse_transform(data))
    # 打印输出得到的数值化数据
    print(data)

    return None


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

    print(cv.get_feature_names())

    print(data.toarray())

    return None





def cutword():

    # 使用jieba对中文进行分词并转换为字符串格式

    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 吧列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3



def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None



def tfidfvec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return: NOne
    """
    # 计算公式：X' = ( X - min )/( max - min )      X'' = X' * ( mx - mn ) + min   其中X''为最终结果；mx和mi分别为指定的区间的最大值和最小值

    #缺点：异常点对最大值最小值影响太大
    # 所以这种方法鲁棒性（稳定性）太差，只适合精确小数据场景

    #feature_range  指定归一化的范围，默认从0到1    数据格式为二维数组格式
    mm = MinMaxScaler(feature_range=(2, 3))

    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])

    print(data)

    return None


def stand():
    """
    标准化缩放
    :return:
    """
    std = StandardScaler()
    # 特点：比归一化稳定，同样 作用于每一列，不容易受到异常点的影响

    # 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景

    # 通过对原始数据进行变换把数据变换到均值为0，标准差为1的范围内

    # 公式为 x' = (x-mean)/e  作用于每一列，mean为平均值，e为标准差（用于考量数据的稳定性）
    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])

    print(data)

    return None


def im():
    """
    缺失值处理
    :return:NOne
    """
    # NaN, nan

    # 实例化Imputer
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # missing_values为指定缺失值，strategy为指定填补方式，此处填补方式为平均数填补，axis为按行或者按列处理，此处为按列处理

    # 关于np.nan(np.NaN)
    # numpy的数组中可以使用np.nan / np.NaN来代替缺失值，属于float类型
    # 如果是文件中的一些缺失值，可以替换成nan，通过np.array转化成float型的数组即可

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)

    return None


# 降维：减少特征的数量
def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """
    var = VarianceThreshold(threshold=0.0)    #threshold指定方差的值，默认是0.0，就是把相同的数据集删除

    # 返回值：训练集差异低于threshold的特征将被删除

    # 默认值是保留所有非零方差特征，及删除所有样本中具有相同值的特征
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)
    return None

#目的：是数据尾数压缩，尽可能降低原数据的维数（复杂度）。损失少量信息
# 作用：可以削减回归分析或者聚类分析中特征的数量

# 当特征数量达到上百的时候要考虑一下数据的简化，数据会发生改变，特征数量也会减少
def pca():
    """
    主成分分析进行特征降维
    :return: None
    """

    #参数可以是小数和整数，如果是小数的话一般位于0到1之间，为百分比格式，意思是要保留百分之多少的信息；
    #如果是整数的话，是保留几个特征值
    pca = PCA(n_components=0.75)

    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])

    # 返回值是指定维度的array
    print(data)

    return None


if __name__ == "__main__":
    pca()