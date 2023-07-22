from funcs.Simulated_annealing import simulated_annealing
from methods.SeqSearching.TimeWindowsBuilding import window
from methods.SeqSearching.Seq2Hw import seq2hw
from methods.SeqSearching.Hw2Sa import hw2sa
from methods.AnsGetting.BestAnsFinding import bestAnsFinding
from methods.AnsGetting.LSTM_Training_Forecasting import lstm_process
import init

def main():
    global flag,solutionSpace, rmseArray, iterOfAnnealing
    global numTimeStepsTrain, numHiddenUnits, learning_rate, num_epochs
    init.initialize1()
    while True:
        seq = window()
        ts = seq2hw(seq)
        flag = hw2sa(seq,ts)
        if flag == False:
            simulated_annealing()
            continue
        else:break
    bestSeq = bestAnsFinding()
    init.initialize2(bestSeq)
    lstm_process(bestSeq)





if __name__ == "__main__":
    main()


'''
examples:

init_holtwinters()  # 初始化Holt-Winters相关全局变量
model_hw = HoltWintersModel(hw_data, hw_n_year, hw_n_predYear, hw_slen, hw_alpha, hw_beta, hw_gamma)
model_hw.show1()

init_grey()  # 初始化灰色预测相关全局变量
predicted_sequence_gm = grey_prediction_with_buffer(gm_data, gm_prediction_length, gm_alpha)
print(predicted_sequence_gm)

'''

