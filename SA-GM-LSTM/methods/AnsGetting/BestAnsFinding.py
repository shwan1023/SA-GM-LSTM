def bestAnsFinding():
    global T,K,solutionSpace, rmseArray, iterOfAnnealing
    try:
        '''
        搜索最优解，即在以迭代轮次为自变量的误差曲线rmseArray中找到最小值和对应的索引并记录，并return最优解给main.py执行LSTM
        '''
        numOfBestIndex = 0
        numOfBestValue = float('inf')

        for i in range(1, iterOfAnnealing + 1):
            if rmseArray[i] < numOfBestValue:
                numOfBestIndex = i
                numOfBestValue = rmseArray[i]

        bestSeq = solutionSpace[numOfBestIndex, :]
        print("Best Turns is: " + str(numOfBestIndex))
        return bestSeq

    except IndexError:
        print("Error: The value of numOfBestIndex is 0.")