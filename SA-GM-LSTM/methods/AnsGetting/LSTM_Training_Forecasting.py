import tensorflow as tf
import matplotlib as plt
import numpy as np
import sys

from funcs.Lstm_forecast import lstm_forecast

sys.path.append('../..')
from funcs import *

def lstm_process(data):
    global numTimeStepsTrain, numHiddenUnits, learning_rate, num_epochs
    lstm_forecast(data, numTimeStepsTrain, numHiddenUnits, learning_rate, num_epochs)