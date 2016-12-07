#######
#Created on Dec 6, 2016
#@author: Minglin Ma
#######

from numpy import *
import matplotlib.pyplot as plt

####
# 数据导入模块
####
def loadData(filename):
	dataset = []
	labels = []
	# fr成了一个打开文件的对象
	fr = open(filename)
	# 这个对象有个readlines函数，一次读取文件所有内容，并返回一个list，每个元素一行
	lines = fr.readlines()
	# 对每一行进行for循环操作
	for line in lines:
		line = line.strip().split()
		# y=w1*x1 + w2*x2 + ... + wm*xm + b 
		# y=w1*x1 + w2*x2 + ... + wm*xm + w0*x0 (x0=1.0,w0=b)
		# 求w和b转换为只求w
		dataset.append([1.0,float(line[0]),float(line[1])])
		labels.append(int(line[2]))
	return dataset,labels
####
# sigmod函数
####
def sigmod(inx):
	return 1.0/(1 + exp(-inx))

####
# 批梯度下降主算法
####
def gradAcent(dataset,labels):
	# list的数组变为mat,以便后面矩阵运算
	dataMat = mat(dataset)
	labelMat = mat(labels).transpose()
	m,n = shape(dataMat)
	# 设置步长
	alpha = 0.001
	# 设置梯度下降次数
	ncycle = 500
	# 初始化W全为1的列向量
	weigh = ones((n,1))
	for i in range(ncycle):
		# 当前weigh拟合的y
		y = dataMat * weigh
		# sigmod压缩
		ey = sigmod(y)
		# 下面涉及是一系列公式推导的结果
		error = (labelMat - ey)
		weigh = weigh + alpha * dataMat.transpose() * error
	return weigh

####
# 用matplotlib画出决策边界
###
def plotBestFit(weights,dataMat,labelMat):
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


####
# 测试
####
dataset,labels = loadData("testSet.txt")
weigh = gradAcent(dataset,labels)
print(weigh)
plotBestFit(weigh.getA(),dataset,labels)