from numpy import *
import operator

# 创建训练数据
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return  group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    ###生成矩阵和计算矩阵
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

def file2matrix(filename):
    fr = open(filename)
    arraylines = fr.readlines()
    numberOfLines = len(arraylines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arraylines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector



if __name__ == "__main__":
    group,labels = creatDataSet()
    a = classify0([1,2],group,labels,3)
    print(a)

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    