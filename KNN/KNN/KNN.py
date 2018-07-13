from numpy import *
import operator

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

if __name__ == "__main__":
	group,labels = createDataSet()
	print(classify0([0,0], group, labels,3))
	print("main")

    