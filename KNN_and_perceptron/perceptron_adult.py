import numpy as np
import pandas as pd
import csv
import copy


# col_labels = ['age', 'wordclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
#               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'wage_class']
# train_set = pd.read_csv('./adult.data', header=None)
# train_set.columns = col_labels
# # 训练集中的缺失值都是用 ? 替换的，首先将其移除:
# train_set = train_set.replace(' ?', np.nan).dropna()

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    # 打乱 idx 的排列
    np.random.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test


def dealdata(filename):
    # 从数据集中获得原始数据
    adult_raw = pd.read_csv(filename, header=None)
    # print(len(adult_raw))
    # 添加标题
    adult_raw.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 4: 'education_number',
                              5: 'marriage', 6: 'occupation', 7: 'relationship', 8: 'race', 9: 'sex',
                              10: 'capital_gain', 11: 'apital_loss', 12: 'hours_per_week', 13: 'native_country',
                              14: 'income'}, inplace=True)
    # 清理数据，删除缺失值
    adult_cleaned = adult_raw.dropna()

    # 属性数字化
    '''这个值得好好学习'''
    adult_digitization = pd.DataFrame()
    target_columns = ['workclass', 'education', 'marriage', 'occupation', 'relationship', 'race', 'sex',
                      'native_country',
                      'income']
    for column in adult_cleaned.columns:
        if column in target_columns:
            unique_value = list(enumerate(np.unique(adult_cleaned[column])))
            dict_data = {key: value for value, key in unique_value}
            adult_digitization[column] = adult_cleaned[column].map(dict_data)
        else:
            adult_digitization[column] = adult_cleaned[column]
    # 确认数据类型为int型数据
    # for column in adult_digitization:
    #     adult_digitization[column] = adult_digitization[column].astype(int)
    # adult_digitization.to_csv("data_cleaned.csv")
    # print(len(adult_cleaned))
    # 构造输入和输出
    X = adult_digitization[
        ['age', 'workclass', 'fnlwgt', 'education', 'education_number', 'marriage', 'occupation', 'relationship',
         'race',
         'sex', 'capital_gain', 'apital_loss', 'hours_per_week', 'native_country']]
    Y = adult_digitization[['income']]
    # 查看数据情况 0:22654, 1:7508
    # print(Y.value_counts())
    # （0.7:0.3）构造训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    X_train.to_csv("X_train.csv", index=None)
    X_test.to_csv("X_test.csv", index=None)
    Y_train.to_csv("Y_train.csv", index=None)
    Y_test.to_csv("Y_test.csv", index=None)


# dealdata('./adult.data')

def loadCSV(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    del dataset[0]
    for i in range(len(dataset)):
        dataset[i] = [float(v) for v in dataset[i]]
    return np.mat(dataset)


data_X = loadCSV('X_train.csv')  # shape: (22792, 14)
data_Y = loadCSV('Y_train.csv')  # shape: (22792, 1)
X_test = loadCSV('X_test.csv')  # shape: (22792, 14)
Y_test = loadCSV('Y_test.csv')
w = np.mat(np.zeros([data_X.shape[1], 1]))
b = np.mat(np.zeros([data_X.shape[0], 1]))
eta = 1
# pred = np.sign(data_X * w + b)

def update(x, y):
    global w, b
    w += eta * x.T * y
    b += eta * y

def perceptron(data_X, data_Y):
    global w, b
    flag = 0
    count = 1
    for j in range(100):
        print("running....")
        for i in range(data_Y.shape[0]):
            if np.sign(data_X[i]*w+b[0]) != data_Y[i]:
                count = 1
                update(data_X[i], data_Y[i])
            else:
                count += 1
        if count > len(data_Y):
            flag = 1
        if flag == 1:
            break

def accuracy(X_test, Y_test):
    count = np.sum((np.sign(X_test*w+b[0]) == Y_test))
    return count * 1.0 / Y_test.shape[0]


if __name__ == '__main__':
    perceptron(data_X, data_Y)
    accu = accuracy(X_test, Y_test)
    print(accu)
