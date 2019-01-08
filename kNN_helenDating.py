#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-
from numpy import *
import operator

#将文本记录到转换Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  #得到文件行数
    returnMat = zeros((numberOfLines,3))    #创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #按列求最小值
    maxVals = dataSet.max(0)    #按列求最大值
    ranges = maxVals - minVals
    # normDataSet=(dataSet-minVals)/ranges
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))       #特征值消除，tile()函数将变量复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges,minVals

#k近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # diffMat = inX- dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5                        #距离计算
    sortedDistIndicies = distances.argsort()            #距离排序从小到大
    classCount={}                                       #创建字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1    #字典的运用
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)    #python2 使用classCount.iteritems()
    return sortedClassCount[0][0]

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.1                   #留百分之十作为测试集
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("预测结果：{0},实际结果：{1}".format(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("错误个数：{0}\n正确率为：{1:f}".format(errorCount,1.0 - errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList=['不感兴趣','有那么一丢丢意思','很感兴趣']
    percentTats=float(input("每天多少小时玩游戏？:"))              #如果是python2请使用raw_input
    ffMiles=float(input("每年飞多少公里数？:"))
    iceCream=float(input("每年吃多少公升冰激凌？:"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minValues=autoNorm(datingDataMat)
    inArr=array([percentTats,ffMiles,iceCream])
    classifierResult=classify0(inArr,datingDataMat,datingLabels,3)
    print("海伦对他：{}".format(resultList[classifierResult-1]))

if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
