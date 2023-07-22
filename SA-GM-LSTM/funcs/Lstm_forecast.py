import tensorflow as tf
import matplotlib as plt
import numpy as np

def lstm_forecast(data, numTimeStepsTrain, numHiddenUnits, learning_rate, num_epochs):
    # 数据预处理，将训练数据标准化为具有零均值和单位方差
    mu = np.mean(data)
    sig = np.std(data)
    dataTrain = data[:numTimeStepsTrain + 1]
    dataTest = data[numTimeStepsTrain + 1:]
    dataTrainStandardized = (dataTrain - mu) / sig

    # 输入 LSTM 的时间序列交替一个时间步
    XTrain = dataTrainStandardized[:-1]
    YTrain = dataTrainStandardized[1:]

    # 创建 LSTM 回归网络
    numFeatures = 1
    numResponses = 1

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(numHiddenUnits, input_shape=(None, numFeatures)),
        tf.keras.layers.Dense(numResponses)
    ])

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')

    # 将数据转换为 TensorFlow 张量
    XTrain = tf.expand_dims(XTrain, axis=-1)
    YTrain = tf.expand_dims(YTrain, axis=-1)

    # 训练 LSTM
    model.fit(XTrain, YTrain, epochs=num_epochs, verbose=0)

    # 测试 LSTM
    dataTestStandardized = (dataTest - mu) / sig
    XTest = dataTestStandardized[:-1]
    YTest = dataTest[1:]

    XTest = tf.expand_dims(XTest, axis=-1)

    YPred = model.predict(XTest)
    YPred = YPred.flatten()
    YPred = sig * YPred + mu

    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((YPred - YTest) ** 2))

    # 将预测值与测试数据进行比较
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(YTest)
    plt.plot(YPred, '.-')
    plt.legend(["Observed", "Predicted"])
    plt.ylabel("Loads")
    plt.title("Forecast with Updates")

    plt.subplot(2, 2, 2)
    plt.stem(YPred - YTest)
    plt.xlabel("Days")
    plt.ylabel("Error")
    plt.title("RMSE = {:.2f}".format(rmse))

    plt.figure()
    plt.subplot(2, 2, 3)
    plt.plot(dataTrain[:-1])
    plt.plot(np.arange(numTimeStepsTrain, numTimeStepsTrain + len(dataTest)), [data[numTimeStepsTrain]] + list(YPred), '.-')
    plt.xlabel("Days")
    plt.ylabel("Loads")
    plt.title("Forecast")
    plt.legend(["Observed", "Forecast"])

    plt.subplot(2, 2, 4)
    plt.plot(data)
    plt.xlabel("Days")
    plt.ylabel("Loads")
    plt.title("Daily load")

    # 计算 MAE、SSE 和 MEAP
    mae = np.mean(np.abs(YTest - YPred))
    sse = np.sum((YPred - YTest) ** 2)
    meap = np.mean(np.abs((YTest - YPred) / YTest) * 100)

    print(f"MAE = {mae}")
    print(f"SSE = {sse}")
    print(f"RMSE = {rmse}")
    print(f"MEAP = {meap}")

    plt.show()