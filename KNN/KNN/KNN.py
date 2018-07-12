from numpy import *
import operator

# 创建训练数据
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return  group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        votellabel = labels[sortedDistIndicies[i]]
        classCount[votellabel] = classCount.get(votellabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    group,labels = creatDataSet()
    a = classify0([1,2],group,labels,3)

    print(a)

    