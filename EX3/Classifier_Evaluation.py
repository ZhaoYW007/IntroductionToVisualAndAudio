import numpy as np
from os.path import exists as os_path_exists
from os import makedirs
from librosa import load as librosa_load
from librosa.feature import mfcc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
print("Import finished")


parameter_num = 15  # the number of mfcc's parameters
train_pos_num = 100
train_neg_num = 100
test_num = 100
dataset_root = 'D:/Temp/Matlab/Machine_Learning/dataset'

train_label = np.append(np.ones([100, 1]), np.zeros([100, 1]), axis=0)


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
        y, sr = librosa_load(dataset_root + '/train/positive/' + str(i) + '/audio.wav', sr=48000)  # initial sr=48000
        mfccs = mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            audio_data1[i][j] = np.mean(mfccs[j][:])
            audio_data1[i][j + parameter_num] = np.var(mfccs[j][:])
    print("Audio positive read finished")

    for i in range(0, train_neg_num):
        y, sr = librosa_load(dataset_root + '/train/negative/' + str(i) + '/audio.wav', sr=48000)
        mfccs = mfcc(y=y, sr=sr, n_mfcc=parameter_num)
        for j in range(0, parameter_num):
            audio_data2[i][j] = np.mean(mfccs[j][:])
            audio_data2[i][j + parameter_num] = np.var(mfccs[j][:])
    print("Audio negative read finished")

    for i in range(0, test_num):
        y, sr = librosa_load(dataset_root + '/test/' + str(i) + '/audio.wav', sr=48000)
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


# def split_train(train__data, train__label, rate):
#     train__ = np.append(train__data, train__label, axis=1)
#     train__num = train__.shape[0]
#     col_rand_array = np.arange(train__num)
#     split_tmp = int(rate * train__num)
#     np.random.shuffle(col_rand_array)
#     train__1 = train__[col_rand_array[0:split_tmp], :]
#     train__2 = train__[col_rand_array[split_tmp + 1:train__num - 1], :]
#     train__data1, train__label1 = np.split(train__1, indices_or_sections=(train__1.shape[1] - 1,), axis=1)
#     train__data2, train__label2 = np.split(train__2, indices_or_sections=(train__2.shape[1] - 1,), axis=1)
#     return train__data1, train__data2, train__label1, train__label2
# image_train1, image_train2, image_train_label1, image_train_label2 = split_train(image_train, train_label, 0.7)
# audio_train1, audio_train2, audio_train_label1, audio_train_label2 = split_train(audio_train_data, train_label, 0.7)


# 3.优化各种分类器的参数
# 优化LR
poly_coef = 3
poly = PolynomialFeatures(poly_coef)  # 生成多项式特征
LR_param = {'penalty': ['l2'], 'tol': [0.00001], 'solver': ['newton-cg'], 'max_iter': [100],
            'multi_class': ['ovr'], 'n_jobs': [-1],
            'C': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]}
# Image
image_train_poly = poly.fit_transform(image_train)
image_train_trans = StandardScaler().fit(image_train_poly)
image_train_standardized = image_train_trans.transform(image_train_poly)
LR_model = GridSearchCV(LogisticRegression(), LR_param, cv=10)
LR_model.fit(image_train_standardized, train_label.ravel())
print('Image LR Parameter:', LR_model.best_params_)
print('Image LR Score:', LR_model.best_score_)
# Audio
audio_train_poly = poly.fit_transform(audio_train_data)
audio_train_trans = StandardScaler().fit(audio_train_poly)
audio_train_standardized = audio_train_trans.transform(audio_train_poly)
LR_model.fit(audio_train_standardized, train_label.ravel())
print('Audio LR Parameter:', LR_model.best_params_)
print('Audio LR Score:', LR_model.best_score_)


# 优化SVC
SVC_param = {'kernel': ['poly'], 'C': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
             'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], 'decision_function_shape': ['ovo']}
SVC_model = GridSearchCV(SVC(), SVC_param, cv=10)
# Image
SVC_model.fit(image_train, train_label.ravel())
print('Image SVC Parameter:', SVC_model.best_params_)
print('Image SVC Score:', SVC_model.best_score_)
# Audio
SVC_model.fit(audio_train_data, train_label.ravel())
print('Audio SVC Parameter:', SVC_model.best_params_)
print('Audio SVC Score:', SVC_model.best_score_)


# 优化Decision Tree
DT_param = {'criterion': ['entropy'], 'min_samples_split': range(2, 20)}
DT_model = GridSearchCV(DecisionTreeClassifier(), DT_param, cv=10)
# Image
DT_model.fit(image_train, train_label)
print('Image DT Parameter:', DT_model.best_params_)
print('Image DT Score:', DT_model.best_score_)
# Audio
DT_model.fit(audio_train_data, train_label)
print('Audio DT Parameter:', DT_model.best_params_)
print('Audio DT Score:', DT_model.best_score_)


# 优化Random Forest
RF_param = {'criterion': ['entropy'], 'min_samples_split': range(2, 20), 'n_estimators': range(2, 20)}
RF_model = GridSearchCV(RandomForestClassifier(), RF_param, cv=10)
# Image
RF_model.fit(image_train, train_label.ravel())
print('Image RF Parameter:', RF_model.best_params_)
print('Image RF Score:', RF_model.best_score_)
# Audio
RF_model.fit(audio_train_data, train_label.ravel())
print('Audio RF Parameter:', RF_model.best_params_)
print('Audio RF Score:', RF_model.best_score_)


# 优化神经网络
MLP_tmp = []
for i in [50, 100, 150, 200]:
    MLP_tmp = MLP_tmp + [(i,)]
    for j in [50, 100, 150, 200]:
        MLP_tmp = MLP_tmp + [(i, j)]
        for k in [50, 75, 100]:
            MLP_tmp = MLP_tmp + [(i, j, k)]
MLP_param = {'hidden_layer_sizes': MLP_tmp, 'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs'], 'max_iter': [10000]}
MLP_model = GridSearchCV(MLPClassifier(), MLP_param, cv=10)
# # Image
MLP_model.fit(image_train, train_label.ravel())
print('Image MLP Parameter:', MLP_model.best_params_)
print('Image MLP Score:', MLP_model.best_score_)
# Audio
MLP_model.fit(audio_train_data, train_label.ravel())
print('Audio MLP Parameter:', MLP_model.best_params_)
print('Audio MLP Score:', MLP_model.best_score_)
