import numpy as np

def grey_prediction_with_buffer(input_sequence, prediction_length, alpha):
    """
    含有弱化缓冲算子的灰色预测函数
    输入：
        input_sequence: 输入序列，一个向量
        prediction_length: 预测长度，即需要预测的未来时间步数
        alpha: 弱化缓冲算子的参数，取值范围为 [0, 1]
    输出：
        predicted_sequence: 预测后的总序列
    """
    # 省略灰色预测的具体代码，使用之前的函数
    n = len(input_sequence)
    cumulative_sequence = np.cumsum(input_sequence)
    adjacent_average = np.zeros(n)
    for i in range(1, n):
        adjacent_average[i] = (cumulative_sequence[i] + cumulative_sequence[i - 1]) / 2
    buffer_sequence = np.zeros(n)
    for i in range(2, n):
        buffer_sequence[i] = 2 * adjacent_average[i] - adjacent_average[i - 1] - alpha * (
                    adjacent_average[i] - 2 * adjacent_average[i - 1] + adjacent_average[i - 2])
    B = np.vstack((-buffer_sequence[1:], np.ones(n - 1))).T
    Y = input_sequence[1:]
    parameters = np.linalg.lstsq(B, Y, rcond=None)[0]
    a = parameters[0]
    b = parameters[1]
    predicted_sequence = np.zeros(n + prediction_length)
    predicted_sequence[0] = input_sequence[0]
    for i in range(1, n + prediction_length):
        predicted_sequence[i] = (input_sequence[0] - b) * np.exp(-a * (i - 1)) + b
    return predicted_sequence