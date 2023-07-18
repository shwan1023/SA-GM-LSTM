function [x_new,y_new] = linear_predicion(x_now,y_now)  
    len = length(x_now);
    len_new = len + 1;
    x_bar = mean(x_now);
    y_bar = mean(y_now);
    item1 = 0 ; item2 = 0; item3 = 0; item4 = 0;
    for i = 1 : len
        item1 = item1 + x_now(i) * y_now(i);
        item2 = item2 + len * x_bar * y_bar;
        item3 = item3 + x_now(i) * x_now(i);
        item4 = item4 + len * x_bar * x_bar;
    end
    bias = (item1 * item2) / (item3 * item4);
    k = y_bar - bias * x_bar;
    x_new = x_now;
    x_new(len_new) = x_new(len) + 1; 
    y_new = y_now;
    y_new(len_new) = k * x_new(len) + bias;
end 
