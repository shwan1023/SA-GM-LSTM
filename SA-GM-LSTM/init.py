import math
import scipy.io as sio

# 初始化全局变量
def initialize1():
    """
    初始化全局变量
    """
    # vars of Basis
    global Fourier,PekingTemp
    Fourier = load_mat_file('data/Fourier.mat')
    PekingTemp = load_mat_file('data/PekingTemp.mat')

    # Parameters Initialization
    global a, b, K, Length, TrainLimit, limit1, limit2,flag,T
    flag = False
    a = 0.3
    b = 0.1
    K = 0.9834
    Length = 5
    T = 1
    TrainLimit = 4745
    limit1 = 0.5
    limit2 = 0.03



    # vars of Holt-Winters
    global data_hw, n_year_hw, n_predYear_hw, slen_hw, alpha_hw, beta_hw, gamma_hw
    data_hw = [30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38, 27, 31, 27, 26, 21, 13, 21,
               18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19, 26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32,
               46, 33, 23, 28, 22, 27, 18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32]
    n_year_hw = 13
    n_predYear_hw = 4
    slen_hw = 365
    alpha_hw = 0.0005
    beta_hw = 0.05
    gamma_hw = 0.05



    globals().update(Fourier,PekingTemp)     # 将mat_data转换为全局变量


    #vars of GM


def initialize2(lstm_trainingData):
    # vars of LSTM
    global lstm_numTimeStepsTrain, lstm_numHiddenUnits, lstm_learningRate, lstm_numEpochs
    lstm_numTimeStepsTrain = math.floor(0.775 * len(lstm_trainingData))
    lstm_numHiddenUnits = 96 * 3
    lstm_learningRate = 0.05
    lstm_numEpochs = 300

def load_mat_file(file_path):
    """
    Load a MATLAB .mat file and convert it into a Python dictionary.
    Args:
        file_path (str): Path to the .mat file.
    Returns:
        dict: A Python dictionary containing the data from the .mat file.
    """
    data = sio.loadmat(file_path)
    return data
