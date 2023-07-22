import numpy as np
from ..AnsGetting.BestAnsFinding import bestAnsFinding


'''
代码效果受硬编码制约，但是数据集已至此，不好更改。
下一次优化，数据清洗办法和数据选择是关键。
'''

def hw2sa(seq,timeSeries):

    global T,K,pekingTemp,iterOfAnnealing,popTimes,solutionSpace,rmseArray
    #initTempVars
    temp = 0
    indx = 0
    each = 0
    YPred = np.zeros(4)
    YReal = np.zeros(4)
    each_year_latter_ave = np.zeros(4)

    # 第一个循环,求得每一年的预测平均值
    for i in range(1, 5):
        for j in range(1, 13):
            # Here,the code could be optimized
            k = 156 + (i - 1) * 12 + j
            temp += timeSeries[k - 1]
        YPred[indx] = temp / 12
        indx = indx + 1
        temp = 0

    # 第二个循环,求得测试集的每一年平均值
    for i in range(14, 18):
        for j in range(1, 346):
            # Here,the code could be optimized
            index = 4745 + (i - 14) * 345 + j
            each = each + pekingTemp[index - 1]
        each_year_latter_ave[i - 14] = each / 345
        YReal[i - 14] = each / 345
        each = 0

    #以4为尺度进行误差计算，为退火判定提供服务
    mae = np.mean(np.abs(YReal - YPred))
    sse = np.sum((YPred - YReal) ** 2)
    rmse = np.sqrt(np.mean((YPred - YReal) ** 2))
    meap = np.mean(np.abs((YReal - YPred) / YReal)) * 100
    print(f"""
    iter = {iterOfAnnealing} 
    MAE = {mae} 
    SSE = {sse} 
    RMSE = {rmse} 
    MEAP = {meap} 
    """)

    # 退火弹出判定
    '''
    popTimes : 弹出的迭代次数（不宜过大，否则很难跳出）
    如果当前持续popTimes次模拟退火误差升高，那么认为当前的解已经没有启发搜索的必要，因此直接搜索解空间(跳转到BestAnsFinding.py)最符合的解
    1、将结果输送到解空间，解空间(solutionSpace)反映了各个轮次下(x)的具体处理序列解(y)
    2、将结果输送到误差序列，误差序列(rmseArray)反映了各个轮次下(x)的具体误差效果(y)
    3、判定：如果邻近5次的误差持续增大被视作停止搜索，进入bestAnsFinding函数
    '''
    for iteriter in range(len(seq)):
        solutionSpace[iterOfAnnealing, iteriter] = seq[iteriter]

    if iterOfAnnealing > popTimes:
        for i in range(1, popTimes + 1):
            if rmseArray[iterOfAnnealing - i + 1] < rmseArray[iterOfAnnealing - i]:
                break
        if i == popTimes:
            # return bestAnsFinding(solutionSpace, rmseArray, iterOfAnnealing)
            return True
    return False
