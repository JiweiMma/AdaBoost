import numpy as np
import matplotlib.pyplot as plt

#创建单层决策树的数据集
def loadSimpData():
    #dataMat - 数据矩阵
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    #classLabels - 数据标签
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#单层决策树分类函数
#dataMatrix - 数据矩阵
#dimen - 第dimen列，也就是第几个特征
#threshVal - 阈值
#threshIneq - 标志
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    #retArray - 分类结果
    # 初始化retArray为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 如果小于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#找到数据集上最佳的单层决策树
#dataArr - 数据矩阵
#classLabels - 数据标签
#D - 样本权重
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0;
    #bestStump - 最佳单层决策树信息
    bestStump = {};
    # bestClasEst - 最佳的分类结果
    bestClasEst = np.mat(np.zeros((m, 1)))
    #minError - 最小误差
    #最小误差初始化为正无穷大
    minError = float('inf')
    #遍历所有特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();
        # 找到特征中最小的值和最大值
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况，均遍历。
            # lt:less than，gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化误差矩阵
                errArr = np.mat(np.ones((m, 1)))
                # 分类正确的,赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算误差
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
