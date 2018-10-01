import numpy as np
import operator
#创建数据集与标签
def createDataSet():
    group =np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# k-近邻算法
def classify0(inX,dataSet,labels,k):  #inx用于分类的输入向量，dataset训练样本集，labels标签向量，k用于选择最近邻居的数目
    #计算距离

    dataSetSize = dataSet.shape[0]  #获取数据集的大小
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet  #tile重复inx，（dataset*1维）1表示在列方向，不带参数表示行方向
    sqDiffMat = diffMat**2  #求平方
    sqDistances =sqDiffMat.sum(axis=1)  #按行求和
    distances=sqDistances**0.5  #求开方
    sortedDisIndices = distances.argsort()
    #按距离的递增关系排序，然后提取其对应的索引
    # 返回值是从小到大的索引值 sortedDisIndicies中保存的是distances的编号
    #不是存的distances中的数据

    #选择距离最小的k个点

    #定一个记录类别次数的的字典
    classCount={}
    for i in range(k):
        #取出前k个类别
        voteIlabel =labels[sortedDisIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse =True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]
def file2matrix(filename):
    love_dictionary = {'largeDoses':3,'smallDoses':2,'didntLike':1}
    #打开文件
    fr = open(filename)
    #读取文件
    arrayOLines=fr.readlines()
    #获得文件的行数
    numberOfLines=len(arrayOLines)
    #返回的numpy矩阵，解密完成的数据：numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector=[]
    #行的索引值
    index=0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片
        listFromLine=line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
        return returnMat,classLabelVector