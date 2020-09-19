import numpy as np
import matplotlib.pyplot as plt

#创建单层决策树的数据集
def loadSimpData():
    #dataMat - 数据矩阵
    datMat = np.matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    #classLabels - 数据标签
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

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

if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    showDataSet(dataArr,classLabels)
