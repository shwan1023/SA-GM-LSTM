import numpy as np

# 模拟退火算法函数
def simulated_annealing():
    global T,K,iterOfAnnealing
    T *= K
    iterOfAnnealing += 1