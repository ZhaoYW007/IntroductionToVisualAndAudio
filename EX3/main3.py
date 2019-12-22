import numpy as np
from csv import reader as csv_read
from os.path import exists as os_path_exists
from os import makedirs
from librosa import load as librosa_load
from librosa.feature import mfcc
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # , Normalizer
from sklearn.neural_network import MLPClassifier
print("Import finished")

parameter_num = 15  # the number of mfcc's parameters
train_pos_num = 100
train_neg_num = 100
test_num = 100
dataset_root = 'D:/Temp/Matlab/Machine_Learning/dataset'

train_label = np.array([1]*train_pos_num + [0]*train_neg_num)
test_label = [0]*test_num
csv_list_test = csv_read(open('D:/Temp/Matlab/Machine_Learning/dataset/test_result.csv', 'r', encoding='utf-8'))
csv_list_test = list(csv_list_test)[0]
csv_list_test[0] = csv_list_test[0].strip('\ufeff')
for i in range(test_num):
    test_label[i] = int(csv_list_test[i])
test_label = np.array(test_label)


# 1.图像特征向量读取
image_read_flag = 0
image_train_pos_root = './image_feat/train_pos.npy'
image_train_neg_root = './image_feat/train_neg.npy'
image_test_root = './image_feat/test.npy'
if not os_path_exists('./image_feat/'):
    makedirs('./image_feat/')
    image_read_flag = 1

image_feat1 = np.empty([train_pos_num, 13])
image_feat2 = np.empty([train_neg_num, 13])
image_test = np.empty([test_num, 13])
if image_read_flag:
    for i in range(0, train_pos_num):
        image_feat1[i][:] = np.load(dataset_root + '/train/positive/' + str(i) + '/feat.npy')
    for i in range(0, train_neg_num):
        image_feat2[i][:] = np.load(dataset_root + '/train/negative/' + str(i) + '/feat.npy')
    for i in range(0, test_num):
        image_test[i][:] = np.load(dataset_root + '/test/' + str(i) + '/feat.npy')

    np.save(file=image_train_pos_root, arr=image_feat1)
    np.save(file=image_train_neg_root, arr=image_feat2)
    np.save(file=image_test_root, arr=image_test)

else:
    image_feat1 = np.load(image_train_pos_root)
    image_feat2 = np.load(image_train_neg_root)
    image_test = np.load(image_test_root)

image_train = np.vstack((image_feat1, image_feat2))
print("Image feat finished")


# 2.音频读取-mfcc提取
# 用于在文件中存储mfcc特征提取的临时结果
audio_read_flag = 0  # 1表示需要提取特征，0表示从csv读取已提取好的特征
mfcc_train_root = './mfcc_tmp/mfcc_train.npy'
mfcc_test_root = './mfcc_tmp/mfcc_test.npy'
if not os_path_exists('./mfcc_tmp/'):
    makedirs('./mfcc_tmp/')
    audio_read_flag = 1

audio_train_data = np.empty([train_pos_num + train_neg_num, 2 * parameter_num])
audio_test_data = np.empty([test_num, 2 * parameter_num])

if audio_read_flag == 1:
    audio_data1 = np.empty([train_pos_num, 2 * parameter_num])
    audio_data2 = np.empty([train_neg_num, 2 * parameter_num])
    for i in range(0, train_pos_num):
        y, sr = librosa_load(dataset_root + '/train/positive/' + str(i) + '/audio.wav', sr=16000)  # initial sr=16000
        mfccs = mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            audio_data1[i][j] = np.mean(mfccs[j][:])
            audio_data1[i][j + parameter_num] = np.var(mfccs[j][:])
    print("Audio positive read finished")

    for i in range(0, train_neg_num):
        y, sr = librosa_load(dataset_root + '/train/negative/' + str(i) + '/audio.wav', sr=16000)
        mfccs = mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            audio_data2[i][j] = np.mean(mfccs[j][:])
            audio_data2[i][j + parameter_num] = np.var(mfccs[j][:])
    print("Audio negative read finished")

    for i in range(0, test_num):
        y, sr = librosa_load(dataset_root + '/test/' + str(i) + '/audio.wav', sr=16000)
        mfccs = mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            audio_test_data[i][j] = np.mean(mfccs[j][:])
            audio_test_data[i][j+parameter_num] = np.var(mfccs[j][:])

    audio_train_data = np.append(audio_data1, audio_data2, axis=0)  # 拼接data1和data2
    np.save(file=mfcc_train_root, arr=audio_train_data)
    np.save(file=mfcc_test_root, arr=audio_test_data)
    print("Audio test read finished")

else:
    audio_train_data = np.load(mfcc_train_root)
    audio_test_data = np.load(mfcc_test_root)
    print("Audio read finished")


# 3.图像特征处理-LR
poly_coef = 3
poly = PolynomialFeatures(poly_coef)  # 生成多项式特征
image_train_poly = poly.fit_transform(image_train)
image_trans_train = StandardScaler().fit(image_train_poly)
image_standardizedX_train = image_trans_train.transform(image_train_poly)  # 标准化0.775 正规化0.69

poly = PolynomialFeatures(poly_coef)  # 生成多项式特征
image_test_poly = poly.fit_transform(image_test)
image_trans_test = StandardScaler().fit(image_test_poly)
image_standardizedX_test = image_trans_test.transform(image_test_poly)

image_clf_LR = LogisticRegression(penalty='l2',  # 惩罚项（l1与l2），默认l2
                         dual=False,  # 对偶或原始方法，默认False，样本数量>样本特征\
                         # 的时候，dual通常设置为False
                         tol=0.00001,  # 停止求解的标准，float类型，默认为1e-4。\
                         # 就是求解到多少的时候，停止，认为已经求出最优解
                         C=0.012,  # 正则化系数λ的倒数，float类型，默认为1.0，越小的数值表示越强的正则化。
                         fit_intercept=True,  # 是否存在截距或偏差，bool类型，默认为True。
                         intercept_scaling=1,  # 仅在正则化项为”liblinear”，\
                         # 且fit_intercept设置为True时有用。float类型，默认为1
                         class_weight='balanced',  # 用于标示分类模型中各种类型的权重，\
                         # 默认为不输入，也就是不考虑权重，即为None，balanced为平衡权重，\
                         # 特殊样本越少权重越大，惩罚越大，解决了失衡问题
                         random_state=1,  # 随机数种子，int类型，可选参数，默认为无
                         solver='newton-cg',  # 优化算法选择参数，只有五个可选参数，\
                         # 即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear
                         max_iter=50,  # 算法收敛最大迭代次数，int类型，默认为10。
                         multi_class='ovr',  # 分类方式选择参数，str类型，可选参数为ovr和multinomial，\
                         # 默认为ovr。如果是二元逻辑回归，ovr和multinomial\
                         # 并没有任何区别，区别主要在多元逻辑回归上。
                         verbose=1,  # 日志冗长度，int类型。默认为0。就是不输出训练过程
                         warm_start=False,  # 热启动参数，bool类型。默认为False。
                         n_jobs=-1  # 并行数。int类型，默认为1。1的时候，\
                         # 用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。
                         )
image_clf_LR.fit(image_standardizedX_train, train_label)  # 拟合训练

image_LR_result = image_clf_LR.predict(image_standardizedX_train)
print('Train Set:', image_clf_LR.score(image_standardizedX_train, train_label))
print('Test Set:', image_clf_LR.score(image_standardizedX_test, test_label))
print('LR Result:', image_LR_result)
print('Manual Result', test_label)
print('\n\n')


# 3.5图像特征处理-SVM
image_classifier_SVM = svm.SVC(C=2, kernel='poly', gamma=10, decision_function_shape='ovo')
image_classifier_SVM.fit(image_train, train_label.ravel())
print("Image SVM trained finished")
# 计算svc分类器的准确率
print("训练集：", image_classifier_SVM.score(image_train, train_label))
print("测试集：", image_classifier_SVM.score(image_test, test_label))
print("Classified finished")
# 查看决策函数
image_SVM_result = image_classifier_SVM.predict(image_test)
# print('train_decision_function:\n', image_classifier_SVM.decision_function(image_train))
print('predict_result:\n', image_SVM_result)
print('manual_result:\n', test_label)
print('\n\n')


# 4.语音特征处理-SVM
# 训练svm分类器
audio_classifier = svm.SVC(C=2, kernel='poly', gamma=10, decision_function_shape='ovo')  # ovo:一对一策略
audio_classifier.fit(audio_train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
print("Audio SVM trained finished")
# 计算svc分类器的准确率
print("训练集：", audio_classifier.score(audio_train_data, train_label))
print("测试集：", audio_classifier.score(audio_test_data, test_label))
print("Classified finished")
# 查看决策函数
audio_SVM_result = audio_classifier.predict(audio_test_data)
# print('train_decision_function:\n', audio_classifier.decision_function(audio_train_data))
print('predict_result:\n', audio_SVM_result)
print('manual_result:\n', test_label)
print('\n\n')


# 5.矩阵合成后处理
total_data_train = np.hstack([image_train, audio_train_data])
total_data_test = np.hstack([image_test, audio_test_data])
total_classifier = MLPClassifier(hidden_layer_sizes=(200, 100,), activation='logistic', solver='lbfgs', tol=0.0001,
                                 max_iter=10000)
total_classifier.fit(total_data_train, train_label)
print("Total NW trained finished")
print("训练集：", total_classifier.score(total_data_train, train_label))
print("测试集：", total_classifier.score(total_data_test, test_label))
print("Classified finished")
total_SVM_result = total_classifier.predict(total_data_test)
print('predict_result:\n', total_SVM_result)
print('manual_result:\n', test_label)
print('\n\n')


audio_fault = 0
image_fault = 0
all_fault = 0
result_type = 0
if result_type:
    image_result = image_SVM_result
else:
    image_result = image_LR_result
for i in range(test_num):
    tmp = int(image_SVM_result[i] != test_label[i]) * 2 + int(audio_SVM_result[i] != test_label[i])
    if tmp == 1:
        print('i=', i, '; Image=', image_result[i], '; Audio=', audio_SVM_result[i], '; Result=', test_label[i],
              '; Type: Audio')
        audio_fault += 1
    if tmp == 2:
        print('i=', i, '; Image=', image_result[i], '; Audio=', audio_SVM_result[i], '; Result=', test_label[i],
              '; Type: Image')
        image_fault += 1
    if tmp == 3:
        print('i=', i, '; Image=', image_result[i], '; Audio=', audio_SVM_result[i], '; Result=', test_label[i],
              '; Type: All')
        all_fault += 1
print('Image Faults:', image_fault)
print('Audio Faults:', audio_fault)
print('All Faults:', all_fault)
