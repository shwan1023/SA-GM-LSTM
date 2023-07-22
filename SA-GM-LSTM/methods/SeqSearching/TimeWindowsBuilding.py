import random
import numpy as np

from funcs.Grey_prediction_with_buffer import grey_prediction_with_buffer
from ...funcs.Linear import linear_prediction
from ... import init


def window():
    # Initialization
    global Fourier, PekingTemp, a, b, K, Length, TrainLimit, limit1, limit2, T , gm_alpha
    originTemp = PekingTemp
    curT = K * T

    # Preallocation
    u = np.zeros(TrainLimit)
    res = np.zeros(TrainLimit)
    aaa = np.zeros(TrainLimit)
    Seq = np.zeros(TrainLimit)

    # Main computation loop
    index = 1
    iterOfWindow = 0
    re_P = []
    while index + Length < TrainLimit:
        '''
        进入时间窗，首先先判断当前的时间窗特征空间
        '''
        iterOfWindow += 1
        deltaY = np.abs(originTemp[index - 1:index + Length - 1] - Fourier[index - 1:index + Length - 1])
        u[iterOfWindow - 1] = np.sum(deltaY ** 2)

        # 特判 时间窗序号为 1
        if iterOfWindow == 1:
            '''
            如果时间窗序号不大于4，那么：
                seq直接取temp（不改变seq）
                第i个时间窗的res直接取第i-1个特征u
                时间窗仅向前移动一个时间步，这是因为没有进入修复模型
            '''
            Seq[index - Length:index] = originTemp[index - Length:index]
            res[iterOfWindow - 1] = u[iterOfWindow - 1]
            index = index - Length + 1
            continue

        '''
        下面的时间窗序号都将大于2，当时间窗介于2~4之间时因为序号过小，通过u和线性回归求得res，否则通过u和灰色预测求得res
        '''
        if u[iterOfWindow] <= res[iterOfWindow - 1]:
            '''
            如果当前特征空间值u小于启发值res，那么证明当前特征空间正常，不做修改(总是拿第i个u与第i-1个res相比较）
            但考虑到也可能不正常（为了跳出局部最优），因此随后还有模拟退火的判定，故1-P应当执行的是类似else的代码，这里需要修改
                第i个时间窗的res取特征空间u线性回归一个时间步后的值（不应当包括1-P的情况）
                ** 这边需要修改if逻辑
            '''
            P = np.exp(-abs(np.tanh(u[iterOfWindow - 1] - u[iterOfWindow - 2])) / curT)
            re_P.append(P)
            if random.random() <= P:
                '''
                如果命中概率P，则进入模拟退火，但是这里模拟退火好像没有写好
                    seq直接取temp（不改变seq）
                    时间窗仅向前移动一个时间步，这是因为没有进入修复模型
                '''
                # 补充1-p，整个代码需要大调整
                Seq[index - Length:index] = originTemp[index - Length:index]
                index = index - Length + 1
            res[iterOfWindow] = linear_prediction(np.arange(1, len(u[:iterOfWindow]) + 1), u[:iterOfWindow])  \
                             if iterOfWindow <= 4 else \
                             res[iterOfWindow] = grey_prediction_with_buffer(u[:iterOfWindow], len(u[:iterOfWindow]) + 1, gm_alpha)
        else:
            '''
            进入修复模型：
                1、求出误差，并判定误差与界的关系
                2、存在界内则不变，否则调整参数a并进行修复，修复过程尽可能令seq靠近temp
                3、时间窗向前移动一个时间窗长度，这是因为已经进入修复模型，时间窗内数据无需再修复
            '''
            ERR = deltaY / originTemp[index - Length: index]
            a = np.where(ERR > limit1, a * (1 + b * abs(np.tanh(ERR - limit1))), a)
            a = np.where(ERR < limit2, np.maximum(a * (1 - b * abs(np.tanh(ERR - limit2))), 0.01), a)
            aaa[:iterOfWindow] = np.where(a < 10, a, aaa[:iterOfWindow])
            gamma = np.minimum(a * deltaY, deltaY)
            Seq[index - Length: index] = np.where(
                originTemp[index - Length: index] > Fourier[index - Length: index],
                originTemp[index - Length: index] - gamma, originTemp[index - Length: index] + gamma)

            res[iterOfWindow] = linear_prediction(np.arange(1, len(u[:iterOfWindow]) + 1), u[:iterOfWindow])  \
                             if iterOfWindow <= 4 else \
                             res[iterOfWindow] = grey_prediction_with_buffer(u[:iterOfWindow], len(u[:iterOfWindow]) + 1, gm_alpha)

            index += Length

    Seq[4745:index - Length - 1] = originTemp[4745:index - Length - 1]

    return Seq
