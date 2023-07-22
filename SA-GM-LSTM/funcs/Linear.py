def linear_prediction(x_now, y_now):
    len_now = len(x_now)
    len_new = len_now + 1

    x_bar = sum(x_now) / len_now
    y_bar = sum(y_now) / len_now

    item1 = sum(x_now[i] * y_now[i] for i in range(len_now))
    item2 = len_now * x_bar * y_bar
    item3 = sum(x_now[i] * x_now[i] for i in range(len_now))
    item4 = len_now * x_bar * x_bar

    bias = (item1 * item2) / (item3 * item4)
    k = y_bar - bias * x_bar

    x_new = x_now.copy()
    x_new.append(x_new[len_now - 1] + 1)

    y_new = y_now.copy()
    y_new.append(k * x_new[len_now] + bias)

    return y_new
