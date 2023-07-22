import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

# Holt-Winters季节性预测类
class HoltWintersModel:
    """
    Perform Holt-Winters prediction.
    """

    def __init__(self, data, n_year, n_predYear, slen, alpha, beta, gamma):
        """
        Holt-Winters模型类
        输入：
            data: 输入数据
            n_year: 年数
            n_predYear: 预测的年数
            slen: 季节性长度
            alpha: Holt-Winters模型的alpha参数
            beta: Holt-Winters模型的beta参数
            gamma: Holt-Winters模型的gamma参数
        """
        self.data = np.array(data)
        self.n_year = n_year
        self.n_predYear = n_predYear
        self.slen = slen
        self.n_pred = self.slen * self.n_predYear
        self.b0 = np.mean([(self.data[i + self.slen] - self.data[i]) / self.slen for i in range(self.slen)])
        ap = [np.mean(self.data[self.slen * (j - 1):self.slen * j]) for j in range(self.n_year)]
        ym = np.array(
            [self.data[i + (j - 1) * self.slen] / ap[j] for j in range(self.n_year) for i in range(self.slen)]).reshape(
            self.slen, self.n_year)
        self.I = [np.mean(ym[i]) for i in range(self.slen)]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.holtwinters()

    def holtwinters(self):
        params_opt, _ = fmin(lambda p: self._holtwinters(p)[0], (self.alpha, self.beta, self.gamma))
        self.ts = self._holtwinters(params_opt)[1]

    def show1(self):
        """
        可视化Holt-Winters模型结果
        """
        plt.figure()
        plt.plot(self.data)
        plt.plot(self.ts, '--r')
        plt.show()

    def _holtwinters(self, params):
        """
        Holt-Winters模型实现
        """
        alpha, beta, gamma = params
        Is = self.I.copy()
        ts = []
        for i in range(len(self.data) + self.n_pred):
            if i == 0:
                smooth = self.data[0]
                trend = self.b0
                ts.append(smooth)
            elif i >= len(self.data):
                m = i - len(self.data) + 1
                ts.append((smooth + m * trend) + Is[(i - 1) % self.slen])
            else:
                val = self.data[i]
                last_smooth, smooth = smooth, alpha * (val - Is[(i - 1) % self.slen]) + (1 - alpha) * (smooth + trend)
                trend = beta * (smooth - last_smooth) + (1 - beta) * trend
                Is[(i - 1) % self.slen] = gamma * (val - smooth) + (1 - gamma) * Is[(i - 1) % self.slen]
                ts.append(smooth + trend + Is[(i - 1) % self.slen])
        return sum((self.data[:len(self.data)] - ts[:len(self.data)]) ** 2), ts


def seq2hw(fy):
    # 数据集长度/季节长度
    global n_year,n_predYear,slen
    # 假设 fy 是一个包含数据的列表或数组
    fy = np.array(fy)

    # 将按天计数的数据转换为按月计数的数据
    data = []
    for i in range(n_year):
        for j in range(slen):
            index_month = i * 12 + j
            temp = 0
            for k in range(30):
                index_day = i * 365 + j * 30 + k
                temp += fy[index_day]
            data.append(temp / 30)

    # 初始化一些用于 Holt-Winters 方法的参数
    model = HoltWintersModel(data, n_year, n_predYear, slen, 0.05, 0.05, 0.05)
    '''
    
    '''

    # 显示预测结果
    model.show1()