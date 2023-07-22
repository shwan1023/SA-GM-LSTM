import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 优化目标函数
def objective_function(x):
    return (x - 2) ** 2 + 5

# 模拟退火算法函数（添加更多控制参数）
def simulated_annealing(lower_bound, upper_bound, initial_temperature, cooling_rate, iterations, alpha=0.95, beta=1):

    """
        模拟退火算法函数
        输入：
            lower_bound: 解空间下界
            upper_bound: 解空间上界
            initial_temperature: 初始温度
            cooling_rate: 退火降温速率
            iterations: 迭代次数
            alpha: 温度下降因子（默认为0.99）
            beta: 步长缩放因子（默认为0.1）
        输出：
            x_min: 最小解
            f_min: 最小目标函数值
            f_history: 目标函数值的历史记录
        """
    x = (upper_bound - lower_bound) * np.random.random() + lower_bound
    f_current = objective_function(x)
    x_min = x
    f_min = f_current
    f_history = [f_current]
    for i in range(iterations):
        step = beta * (np.random.random() - 0.5)
        x_new = x + step
        x_new = max(x_new, lower_bound)
        x_new = min(x_new, upper_bound)
        f_new = objective_function(x_new)
        delta_e = f_new - f_current
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / initial_temperature):
            x = x_new
            f_current = f_new
        if f_current < f_min:
            x_min = x
            f_min = f_current
        initial_temperature = initial_temperature * cooling_rate
        beta *= alpha  # 步长逐渐缩小
        f_history.append(f_current)
    return x_min, f_min, f_history

# 设置解空间下界和上界
lower_bound = -10
upper_bound = 10

# 设置初始温度、退火降温速率和迭代次数
initial_temperature = 10000
cooling_rate = 0.97
iterations = 100

# 运行模拟退火算法并获取历史目标函数值
x_min, f_min, f_history = simulated_annealing(lower_bound, upper_bound, initial_temperature, cooling_rate, iterations)

print("最小解 x_min:", x_min)
print("最小目标函数值 f_min:", f_min)

# 可视化展示退火过程
plt.plot(f_history)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('模拟退火算法优化过程')
plt.show()
