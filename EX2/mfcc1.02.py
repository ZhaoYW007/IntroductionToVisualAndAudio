import numpy as np
import librosa
import csv
from sklearn import svm
import os
# from sklearn.model_selection import train_test_split

parameter_num = 15  # the number of mfcc's parameters
train_pos_num = 100
train_neg_num = 100
test_num = 100
dataset_root = 'D:/Temp/Matlab/Machine_Learning/dataset'

# 用于在文件中存储mfcc特征提取的临时结果
read_flag = 0  # 1表示需要提取特征，0表示从csv读取已提取好的特征
mfcc_train_pos_root = './mfcc_tmp/mfcc_train_pos.npy'
mfcc_train_neg_root = './mfcc_tmp/mfcc_train_neg.npy'
mfcc_test_root = './mfcc_tmp/mfcc_test.npy'
if not os.path.exists('./mfcc_tmp/'):
    os.makedirs('./mfcc_tmp/')
    read_flag = 1

data1 = np.empty([train_pos_num, 2 * parameter_num + 1])
data2 = np.empty([train_neg_num, 2 * parameter_num + 1])
data3 = np.empty([test_num, 2 * parameter_num + 1])

if read_flag == 1:
    for i in range(0, train_pos_num):
        y, sr = librosa.load(dataset_root + '/train/positive/' + str(i) + '/audio.wav', sr=16000)  # initial sr=16000
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            data1[i][j] = np.mean(mfccs[j][:])
            data1[i][j + parameter_num] = np.var(mfccs[j][:])

    print("stage one finished")

    for i in range(0, train_neg_num):
        y, sr = librosa.load(dataset_root + '/train/negative/' + str(i) + '/audio.wav', sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            data2[i][j] = np.mean(mfccs[j][:])
            data2[i][j + parameter_num] = np.var(mfccs[j][:])

    print("stage two finished")

    for i in range(0, test_num):
        y, sr = librosa.load(dataset_root + '/test/' + str(i) + '/audio.wav', sr=16000)
        mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = parameter_num)
        for j in range(0, parameter_num):
            data3[i][j] = np.mean(mfccs[j][:])
            data3[i][j+parameter_num] = np.var(mfccs[j][:])

    for i in range(0, train_pos_num):
        data1[i][2 * parameter_num] = 1

    for i in range(0, train_neg_num):
        data2[i][2 * parameter_num] = 0

    csv_list = csv.reader(open('D:/Temp/Matlab/Machine_Learning/dataset/test_result.csv', 'r', encoding='utf-8'))
    csv_list = list(csv_list)[0]
    csv_list[0] = csv_list[0].strip('\ufeff')
    for i in range(test_num):
        data3[i][2 * parameter_num] = int(csv_list[i])

    np.save(file=mfcc_train_pos_root, arr=data1)
    np.save(file=mfcc_train_neg_root, arr=data2)
    np.save(file=mfcc_test_root, arr=data3)
    print("stage 2.5 finished")

else:
    data1 = np.load(mfcc_train_pos_root)
    data2 = np.load(mfcc_train_neg_root)
    data3 = np.load(mfcc_test_root)


data = np.append(data1, data2, axis=0)  # 拼接data1和data2
# 1.划分数据与标签
train_data, train_label = np.split(data, indices_or_sections=(parameter_num * 2,), axis=1)  # x为数据，y为标签
test_data, test_label = np.split(data3, indices_or_sections=(parameter_num * 2,), axis=1)
x2, y2 = np.split(data2, indices_or_sections=(parameter_num * 2,), axis=1)
print("stage three finished")

# 2.训练svm分类器
classifier = svm.SVC(C=2, kernel='poly', gamma=10, decision_function_shape='ovo')  # ovo:一对一策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
print("stage four finished")

# 3.计算svc分类器的准确率
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))
print("测试集2:", classifier.score(x2, y2))
print("stage five finished")
# 4.查看决策函数
print('train_decision_function:\n', classifier.decision_function(train_data))
print('predict_result:\n', classifier.predict(test_data))
print('predict_result2:\n', classifier.predict(x2))

