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
    # classLabels - 数据标签
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#数据可视化
#dataMat - 数据矩阵
#labelMat - 数据标签
def showDataSet(dataMat, labelMat):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    # 负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

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

# 找到数据集上最佳的单层决策树
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
    #bestClasEst - 最佳的分类结果
    bestClasEst = np.mat(np.zeros((m, 1)))
    #minError - 最小误差
    # 最小误差初始化为正无穷大
    minError = float('inf')
    # 遍历所有特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();
        # 找到特征中最小的值和最大值
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况，均遍历。lt:less than，gt:greater than
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
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

#使用AdaBoost算法提升弱分类器性能
#dataArr - 数据矩阵
#classLabels - 数据标签
#numIt - 最大迭代次数
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    #weakClassArr - 训练好的分类器
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 初始化权重
    D = np.mat(np.ones((m, 1)) / m)
    #aggClassEst - 类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:",D.T)
        # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        # print("classEst: ", classEst.T)
        # 计算e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        # 根据样本权重公式，更新样本权重
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # 计算类别估计累计值
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        # 计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        # 误差为0，退出循环
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst

#AdaBoost分类函数
#datToClass - 待分类样例
#classifierArr - 训练好的分类器
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 遍历所有分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print(adaClassify([[0, 0], [5, 5]], weakClassArr))
