from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	#计算当前数据总数。
	dataSetSize = dataSet.shape[0]
	#计算个点与输入点的距离
	#
	#distance**2 = （x1 - x2）**2 + (y1 - y2)**2
	#dataSetSize 列
	#[inX, inY]
	#[inX, inY]  
	#[inX, inY] - 原有矩阵得到其差值。
	#[inX, inY]
	#
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistance = sqDiffMat.sum(axis=1)
	distance = sqDistance**0.5
	sortedDistance = distance.argsort()
	classCount = {}

	for i in range(k):
		votelLabel = labels[sortedDistance[i]]
		classCount[votelLabel] = classCount.get(votelLabel,0)+1
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
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m ,1))

    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('F:\Code\Python\KNN\KNN\datingTestSet.txt')
    normdatingDataMat,ranges, minVals = autoNorm(datingDataMat)
    m = normdatingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)

    errCount = 0.0

    for i in range(numTestVecs):
        classiFierResult= classify0(normdatingDataMat[i,:], normdatingDataMat[numTestVecs:m,:], \
        datingLabels[numTestVecs:m],3)
        #print ("the classifier came back with: %s, the real answer is: %s" %(classiFierResult, datingLabels[i]))
        if(classiFierResult != datingLabels[i]):
            errCount += 1.0
            print ("the classifier came back with: %s, the real answer is: %s" %(classiFierResult, datingLabels[i]))

    print ("the total error rate is: %f" %(errCount/float(numTestVecs)))



if __name__ == "__main__":
<<<<<<< HEAD
	group,labels = createDataSet()
	print(classify0([0,0], group, labels,3))
    
=======
    group, labels = creatDataSet()
    a = classify0([1, 2], group, labels, 3)
    #print(a)
    datingDataMat, datingLabels = file2matrix('F:\Code\Python\KNN\KNN\datingTestSet2.txt')
    #print(datingDataMat)
    #print(datingLabels)

    normdatingDataMat,ranges,minVals = autoNorm(datingDataMat)
    #print(datingDataMat)
    #print(ranges)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(normdatingDataMat[:, 0], normdatingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    #plt.show()

    datingClassTest()



>>>>>>> 6fd19e6c2c6d9e1c7049deb5d0bc1c31d85c44a2
