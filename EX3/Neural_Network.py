import numpy as np
from csv import reader as csv_read
from os.path import exists as os_path_exists
from os import makedirs
from librosa import load as librosa_load
from librosa.feature import mfcc
from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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


# 3. classify image
poly_coef = 3
poly = PolynomialFeatures(poly_coef)  # 生成多项式特征
image_train_poly = poly.fit_transform(image_train)
image_trans_train = StandardScaler().fit(image_train_poly)
image_standardizedX_train = image_trans_train.transform(image_train_poly)  # 标准化0.775 正规化0.69

poly = PolynomialFeatures(poly_coef)  # 生成多项式特征
image_test_poly = poly.fit_transform(image_test)
image_trans_test = StandardScaler().fit(image_test_poly)
image_standardizedX_test = image_trans_test.transform(image_test_poly)
image_result = np.array([0]*test_num)
irratation_time1 = 10001
for i in range(irratation_time1):
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
                             random_state=None,  # 随机数种子，int类型，可选参数，默认为无
                             solver='newton-cg',  # 优化算法选择参数，只有五个可选参数，\
                             # 即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear
                             max_iter=50,  # 算法收敛最大迭代次数，int类型，默认为10。
                             multi_class='ovr',  # 分类方式选择参数，str类型，可选参数为ovr和multinomial，\
                             # 默认为ovr。如果是二元逻辑回归，ovr和multinomial\
                             # 并没有任何区别，区别主要在多元逻辑回归上。
                             verbose=0,  # 日志冗长度，int类型。默认为0。就是不输出训练过程
                             warm_start=False,  # 热启动参数，bool类型。默认为False。
                             n_jobs=-1  # 并行数。int类型，默认为1。1的时候，\
                             # 用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。
                             )
    image_clf_LR.fit(image_standardizedX_train, train_label)  # 拟合训练
    tmp_result = image_clf_LR.predict(image_standardizedX_train)
    for j in range(test_num):
        if tmp_result[j]:
            image_result[j] += 1


# 4. classify audio
irratation_time2 = 10001
score = 0
total_data_train = audio_train_data
total_data_test = audio_test_data
total_result = np.array([0] * test_num)
result = np.array([0] * test_num)
for i in range(irratation_time2):
    total_classifier = DecisionTreeClassifier(min_samples_split=4)
    total_classifier.fit(total_data_train, train_label)
    tmp_result = total_classifier.predict(total_data_test)
    for j in range(test_num):
        if tmp_result[j]:
            total_result[j] += 1
print('predict_result:\n', total_result)
for i in range(test_num):
    if total_result[i] > 2*(irratation_time2-1)/3:
        result[i] = 1
    elif total_result[i] < (irratation_time2-1)/3:
        result[i] = 0
    else:
        result[i] = int(image_result[i] > (irratation_time1-1)/2)
    if result[i] == test_label[i]:
        score += 1
    else:
        print(i, result[i], test_label[i])
print("Classified finished")
print('predict_result:\n', result)
print('manual_result:\n', test_label)
print('Score:', score)
print('\n\n')

# param_range = range(2, 50)
# train_score, test_score = validation_curve(total_classifier, total_data_train, train_label,
#                                            param_name='C', param_range=param_range,
#                                            cv=20, scoring='accuracy')
# train_score = np.mean(train_score, axis=1)
# test_score = np.mean(test_score, axis=1)
# plt.plot(param_range, train_score, 'o-', color='r', label='training')
# plt.plot(param_range, test_score, 'o-', color='g', label='testing')
# plt.legend(loc='best')
# plt.xlabel('Value')
# plt.ylabel('accuracy')
# plt.show()

