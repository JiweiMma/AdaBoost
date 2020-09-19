import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat

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

#使用AdaBoost算法训练分类器
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

#绘制ROC
#predStrengths - 分类器的预测强度
#classLabels - 类别
def plotROC(predStrengths, classLabels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 绘制光标的位置
    cur = (1.0, 1.0)
    # 用于计算AUC
    ySum = 0.0
    # 统计正类的数量
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    # y轴步长
    yStep = 1 / float(numPosClas)
    # x轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)

    # 预测强度排序,从低到高
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;
            delY = yStep
        else:
            delX = xStep;
            delY = 0
            # 高度累加
            ySum += cur[1]
        # 绘制ROC
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # 更新绘制光标的位置
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    # 计算AUC
    print('AUC面积为:', ySum * xStep)
    plt.show()


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('D:\horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 10)
    plotROC(aggClassEst.T, LabelArr)