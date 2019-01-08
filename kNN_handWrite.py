#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir

#将图像转化为向量，图像32*32转化为向量1*1024
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

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

"""
#手写数字识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of error is :%d"%errorCount)
    print("\nthe total error rate is :%f"%(errorCount/float(mTest)))
handwritingClassTest()
"""
def testhandwritingClassTest():
    """获取目录内容
    从文件名解析分类数字
    """
    labels = []
    training_path = 'trainingDigits'
    test_path = 'testDigits'
    files = listdir(training_path)
    dataSetMat = zeros((len(files),1024))
    index = 0
    errorcount = 0.0
    for filename in files:
        dataSetMat[index,:] = img2vector(training_path+'\\'+filename)
        labels.append(int(filename[0]))
        index += 1
    testfiles = listdir(test_path)
    m = len(testfiles)
    for testfile_name in testfiles:
        test_vector = img2vector(test_path+'\\'+testfile_name)
        test_label = classify0(test_vector,dataSetMat,labels,3)
        print("预测值是:%d,正确值是:%d"%(test_label,int(testfile_name[0])))
        if (test_label != int(testfile_name[0])):
            errorcount += 1.0
    print("\n错误的个数 :%d"%errorcount)
    print("正确率：{0:f}".format(1.0-(errorcount/float(m))))
testhandwritingClassTest()