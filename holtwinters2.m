%% Triple exponential smoothing
% runs holt-winter algorithm
% credits to the fantastic blog post of grisha trubetskoy 
% take a look: https://grisha.org/blog/2016/02/17/triple-exponential-smoothing-forecasting-part-iii/
% still under construction, function can't be optimized

% clear
% clc
% close all
% 
% data = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,...
%          27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,...
%          26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,...
%          18,8,17,21,31,34,44,38,31,30,26,32]; %example data set used by g. trubetskoy
data = 0;
for i = 1 : 4745
    data(i) = PekingTemp(i);
end

hold off
n_year=13; %length of data set
n_predYear = 4;
slen=365; %seasonal length

b0=[];
for i=1:slen
    b0=[b0 (data(i+slen)-data(i))/slen];
end
b0=sum(b0)/slen; % initial trend, here average of trend across seasons

ap=[];
for j=1:n_year
    ap=[ap sum(data(slen*(j-1)+1:slen*j))/slen];
end % averages of each year

ym=zeros(slen,n_year);
for j=1:n_year
    for i=1:slen
        ym(i,j)=data(i+(j-1)*slen)/ap(j);
    end
end % division by appropriate yearly mean
I=[];
for i=1:slen
        I=[I sum(ym(i,:))/n_year];
end % initial seasonal components

n_pred=slen*n_predYear; %forecasting period


% alpha=0.05;
% beta=0.05;
% gamma=0.05;

% alpha=0.005;
% beta=0.0005;
% gamma=0.5;

alpha = 0.0005;
beta = 0.05;
gamma = 0.05;

p=[alpha beta gamma]; %calibration parameters
X0=[p data n_pred b0 I slen]; %initial values
    
options.Display='iter';
options.FunValCheck='on';
options.PlotFcns=@optimplotfval;
numbruns=200*length(p);
options.MaxFunEvals=numbruns;
options.MaxIter=numbruns;

minFunc = @(p)holtwinters(p,data,n_pred,b0,I,slen);
[x,fval,exitflag,output]=fminsearch(minFunc,p,options);

[ee,ts]=holtwinters(x,data,n_pred,b0,I,slen);

% visualisation of results
figure(2)
plot(data)
hold on
plot(ts,'--r')