function[ee,ts]=holtwinters(p,data,n_pred,b0,I,slen)

ts=[];
alpha=p(1);
beta=p(2);
gamma=p(3);
Is=I;
for i=1:length(data)+n_pred
    if i==1
        smooth=data(1);
        trend=b0;
        ts=[ts smooth];
    elseif i>=length(data)
        m=i-length(data)+1;
        ts=[ts (smooth+m*trend)+Is(mod(i-1,slen)+1)];
    else
        val=data(i);
        last_smooth=smooth;
        smooth=alpha*(val-Is(mod(i-1,slen)+1))+(1-alpha)*(smooth+trend);
        trend=beta*(smooth-last_smooth)+(1-beta)*trend;
        Is(mod(i-1,slen)+1)=gamma*(val-smooth)+(1-gamma)*Is(mod(i-1,slen)+1);
        ts=[ts smooth+trend+Is(mod(i-1,slen)+1)];
    end
end
ee=sum((data(1:length(data))-ts(1:length(data))).^2);