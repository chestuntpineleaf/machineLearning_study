from sklearn.datasets import load_iris,load_boston,fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 分类数据集   最后得到的数据是离散型数据
li = load_iris()

print("获取数据数组")  #特征值
print(li.data)  # 特征数据数组，是[n_samples * n_features]的二维numpy.ndarray数组
print("标签数组")   #目标值
print(li.target)  #标签数组，是n_samples 的一维numpy.ndarray数组    0,1,2表示三种花的类别  class:- Iris-Setosa - Iris-Versicolour - Iris-Virginica
print("特征名")
print(li.feature_names)  #标签数组，是n_samples的一维numpy.ndarray数组
print("标签值")
print(li.target_names)  #标签名   sepal萼片   petal 花瓣
print("数据描述")
print(li.DESCR)  #数据描述

#-----------------------------分割线------------------------------
# 对数据集进行分割
#注意返回值，训练集 train，x_train，y_train   测试集  test   x_test  ， y_test     x代表特征值，y代表目标值
x_train,x_test,y_train, y_test = train_test_split(li.data,li.target,test_size=0.25)   #test_size指定测试集的大小
print("训练集特征值和目标值：",x_train,y_train)
print("训练集特征值和目标值：",x_test,y_test)

# 用于分类的大数据集    fetch_20newsgroups提供20类的数据集
news = fetch_20newsgroups(subset='all')   #subset用于选择要加载的数据集，可选"train","test","all"
print(news.DESCR)
print(news.data)
print(news.target)   #获取到每个数对应的类别

#-----------------------------分割线------------------------------
# 回归数据集，目标值都是连续的
lb = load_boston()
print(lb.DESCR)
print("获取特征值")
print(lb.data)   #特征值是房子的各项属性
print("目标值")
print(lb.target)   #连续的

#-----------------------------分割线------------------------------
#fit_transform  数据集的转换     转换器   是一种实现特征工程的api
# fit_transform():输入后直接转换

# fit():输入数据，但是不做事情  [[1,2,3],[4,5,6]]   计算平均值，方差等来获得标准
# transform():进行数据的转换

# fit_transform  =  fit() + transform()

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
s.fit_transform([[1,2,3],[4,5,6]])   #直接转换

ss = StandardScaler()
ss.fit([[1,2,3],[4,5,6]])   #先输入数据，对数据进行计算得出标准
ss.transform([[1,2,3],[4,5,6]])   #用上面的标准（平均值，标准差）对数据进行转换

ss.fit([[2,3,4],[4,5,6]])    #得到的结果与上面不同
ss.transform([[1,2,3],[4,5,6]])

#-----------------------------分割线------------------------------
#估计器  estimator  通过训练集的数据对输入进来的测试集数据进行预测并计算准确率
 
# 1.调用fit()输入训练集数据  fit(x_train,y_train)
# 2.输入测试集的数据进行预测  y_predict = predict (x_test)   先对y的值进行预估   再计算准确率  score（x_test，y_test