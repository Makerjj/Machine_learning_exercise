# -*- coding: utf-8 -*-

import numpy as np
import random
import csv
import math


# 加载csv文件，将所有数据处理成 float 类型的 list 格式，存储在 dataSet 中
def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataSet = list(lines)
    for i in range(len(dataSet)):
        dataSet[i] = [float(x) for x in dataSet[i]]
    return dataSet

# 将 dataSet 分为 trainSet 和 dataSet ，存在一个 list 中， list[0] 为 trainSet
def splitDataSet(dataSet, splitRatio):
   trainSize = int(len(dataSet) * splitRatio)
   trainSet = []
   copy = dataSet
   while len(trainSet) < trainSize:
       index = random.randrange(len(copy))
       trainSet.append(copy.pop(index))
   return [trainSet, copy]

# 按类别划分数据,以 dict 的形式储存
def separateByClass(dataSet):
    separated = {}
    for i in range(len(dataSet)):
        vector = dataSet[i]
        if vector[0] not in separated:
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
    return separated

# 计算每个类别中每个属性的均值
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# 计算每个类别中每个属性的标准差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# 计算出 dataSet 中每个属性的平均值和标准差
# 这里的 dataSet 为已经分好类的，即下文的 instances
def summarize(dataSet):
    summaries = [(np.mean(attribute), stdev(attribute)) for attribute in zip(*dataSet)]
    del summaries[0]
    return summaries

# 首先将数据集按类别分类，存储在 separated 字典中，然后对该字典遍历，
def summarizeByClass(dataSet):
    separated = separateByClass(dataSet)
    summaries = {}
    # instances 为类别为 classValue 的 dataSet 中的所有列表
    # 然后对这些列表中的每个属性计算均值和标准差，存入summaries
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbabobility(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
    print('x:',x, '    ',  exponent)
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# summaries 储存着的是单个样本的每个类别及它们对应着的所有属性的均值和标准差
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    # classSummaries 存储着类别 classValue 中所有属性的均值和标准差
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i + 1]
            # print('mean:',mean, 'stdev:', stdev, '概率', calculateProbabobility(x, mean, stdev), 'probabilities[classValue]:', probabilities[classValue])
            probabilities[classValue] *= calculateProbabobility(x, mean, stdev)
    return probabilities

# 对于单个样本，根据最大的概率值判断其所属的类别
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# 对 test 集进行预测
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

# 将真实值和预测值进行比较，算出accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        print('真实值：{:%<2f}'.format(testSet[x][0]), '预测值：{:%<5f}'.format(predictions[x]))
        if testSet[x][0] == predictions[x]:
            correct += 1
    return correct / float(len(testSet))


dataSet = loadCsv('pima-indians-diabetes.csv')
dataSet = loadCsv('windata.csv')

# dataSet[0] 为trainSet， dataSet[1] 为 testSet
dataSet = splitDataSet(dataSet, 0.6)
separated = separateByClass(dataSet[0])
print(len(dataSet[0][0]))
# summary = summarize(dataSet[0])
# 存储着每个类别的所有属性的均值和标准差
summaries = summarizeByClass(dataSet[0])
predictions = getPredictions(summaries, dataSet[1])
accuracy = getAccuracy(dataSet[1], predictions)
print("accuracy: {:.2%}".format(accuracy))

