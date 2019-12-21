import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.utils import shuffle

data1 = np.empty([100, 13])
data2 = np.empty([100, 13])
data3 = np.empty([100, 13])
for i in range(0, 100):
    data1[i][:] = np.load('E:/dataset/train/positive/' + str(i) + '/feat.npy')
for i in range(0, 100):
    data2[i][:] = np.load('E:/dataset/train/negative/' + str(i) + '/feat.npy')

for i in range(0, 100):
    data3[i][:] = np.load('E:/dataset/test/' + str(i) + '/feat.npy')

x_train = np.vstack((data1, data2))
# x_test = data3
x_train = list(x_train)
y_train = [1] * 100 + [0] * 100

poly = PolynomialFeatures(3)  # 生成多项式特征
x_train_poly = poly.fit_transform(x_train)
# x_test_poly = poly.fit_transform(x_test)
xtrans = StandardScaler().fit(x_train_poly)
# xtest_trans= StandardScaler().fit(x_test_poly)
standaedizedX = xtrans.transform(x_train_poly)  # 标准化0.775 正规化0.69
# standaedizedXtest=xtrans.transform(x_test_poly)

clf = LogisticRegression(penalty='l2',  # 惩罚项（l1与l2），默认l2
                         dual=False,  # 对偶或原始方法，默认False，样本数量>样本特征\
                         # 的时候，dual通常设置为False
                         tol=0.00001,  # 停止求解的标准，float类型，默认为1e-4。\
                         # 就是求解到多少的时候，停止，认为已经求出最优解
                         C=20.0,  # 正则化系数λ的倒数，float类型，默认为1.0，越小的数值表示越强的正则化。
                         fit_intercept=True,  # 是否存在截距或偏差，bool类型，默认为True。
                         intercept_scaling=1,  # 仅在正则化项为”liblinear”，\
                         # 且fit_intercept设置为True时有用。float类型，默认为1
                         class_weight='balanced',  # 用于标示分类模型中各种类型的权重，\
                         # 默认为不输入，也就是不考虑权重，即为None，balanced为平衡权重，\
                         # 特殊样本越少权重越大，惩罚越大，解决了失衡问题
                         random_state=1,  # 随机数种子，int类型，可选参数，默认为无
                         solver='liblinear',  # 优化算法选择参数，只有五个可选参数，\
                         # 即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear
                         max_iter=50,  # 算法收敛最大迭代次数，int类型，默认为10。
                         multi_class='ovr',  # 分类方式选择参数，str类型，可选参数为ovr和multinomial，\
                         # 默认为ovr。如果是二元逻辑回归，ovr和multinomial\
                         # 并没有任何区别，区别主要在多元逻辑回归上。
                         verbose=0,  # 日志冗长度，int类型。默认为0。就是不输出训练过程
                         warm_start=False,  # 热启动参数，bool类型。默认为False。
                         n_jobs=1  # 并行数。int类型，默认为1。1的时候，\
                         # 用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。
                         )
clf.fit(standaedizedX, y_train)  # 拟合训练
print(clf.score(standaedizedX, y_train))
print(clf.predict(standaedizedX))

# print(clf.coef_)#模型参数
# print(clf.predict(standaedizedXtest))

# 过拟合可视化，随着不同数量的输入错误率变化的曲线
# train_sizes,train_score,test_score = learning_curve(clf, standaedizedX, y_train,
#                 train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#                 cv=10, scoring='accuracy')
# train_error = 1 - np.mean(train_score,axis=1)
# test_error = 1 - np.mean(test_score,axis=1)
# plt.plot(train_sizes,train_error,'o-',color = 'r',label = 'training')
# plt.plot(train_sizes,test_error,'o-',color = 'g',label = 'testing')
# plt.legend(loc='best')
# plt.xlabel('traing examples')
# plt.ylabel('error')
# plt.show()
# x_t,x_test,y_t,y_test=train_test_split(standaedizedX,y_train,test_size=0.01)
# print(clf.score(x_t,y_t))
# print(clf.predict(x_t))

# 调参可视化，随着参数不同，正确率变化的曲线
param_range = range(10, 100, 10)
train_score, test_score = validation_curve(clf, standaedizedX, y_train,
                                           param_name='C', param_range=param_range,
                                           cv=20, scoring='accuracy')
train_score = np.mean(train_score, axis=1)
test_score = np.mean(test_score, axis=1)
plt.plot(param_range, train_score, 'o-', color='r', label='training')
plt.plot(param_range, test_score, 'o-', color='g', label='testing')
plt.legend(loc='best')
plt.xlabel('Value')
plt.ylabel('accuracy')
plt.show()