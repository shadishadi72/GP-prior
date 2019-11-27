 % load train and test 
 clear all 
 clc
 load('mean_parameters_w10.mat')
 load('mean_parameters_w5.mat')
load('mean_parameters_dw5.mat')

 data=mean_parameters_dw5;
 % random choice 
% X_train=data([1:200],[2:8]);
% Y_train=data([1:200],9);
% X_test=data([201:end],[2:8]);
% Y_test=data([201:end],9);

[X_train,Y_train,X_test,Y_test] = data_loading_pain_reg(data);
infFun = @infEP; %inference method

train_iter=100;

% s = 5; 
% hyp.mean =[1:s]'; % discrete mean with 12 hypers
% % hyp.cov = randn(s,1);
%  meanfunc= {'meanDiscrete',s};
% meanfunc= {'meanConst'};
 meanfunc= {'meanLinear'};

% hyp.mean=10;

% meanfunc = {@meanSum, {@meanLinear, @meanConst}}
hyp.mean = log(ones(size(X_train,2),1));

covfunc = @covSEard;
% covfunc = @covConst;
ell = 2; sf = 2;
% s=5;
% covfunc = {@covDiscrete,s};
% L = randn(s); L = chol(L'*L); L(1:(s+1):end) = log(diag(L));
% hyp.cov = L(triu(true(s)));



% hyp.cov = log([ones(1,s)*ell, sf]);

hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
% hyp.cov = 10;

likfunc = @likGauss;
% infFun = @infVB; %inference method

sn = 2; 
 hyp.lik = log(sn);
   
%   covfunc = @covSEiso;              % Squared Exponental covariance function
%   likfunc = @likGauss;              % Gaussian likelihood
%     hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
% minimize negative log marginal likelihood 
  hyp2 = minimize(hyp, @gp, -train_iter, @infGaussLik, meanfunc, covfunc, likfunc, X_train, Y_train);
  [mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X_train, Y_train,X_test);
 
%  f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)]; % confidence interval +-2std
%   fill([X_test; flipdim(X_test,1)], f, [7 7 7]/8)
%   hold on; plot(X_test, mu); plot(X_train, Y_train, '+')


% treating as classification 
mu2=zeros(size(mu,1),1);

[c0]=find(mu>=0 & mu <=0.5 );
mu2(c0)=0;

[c1]=find(mu>0.5 & mu <=1 );
mu2(c1)=0;
[c3]=find(mu>1 & mu <=1.5 );
mu2(c3)=1;

[c4]=find(mu>1.5 & mu <=2 );
mu2(c4)=2;
[c5]=find(mu>2 & mu <=2.5 );
mu2(c5)=2;

[c6]=find(mu>2.5 & mu <=3 );
mu2(c6)=3;
[c7]=find(mu>3 & mu <=3.5 );
mu2(c7)=3;

[c8]=find(mu>3.5 & mu <=4);
mu2(c8)=4;
[c9]=find(mu>4);
mu2(c9)=4;



% figure 
% plot(X_train(:,1), Y_train, '+')
% hold on
% plot(X_test, mu,'*r')
% plot(X_test, mu2,'*g')
% hold on
% plot(X_test,Y_test,'*b')


% evalutaion 
predictions=mu2;
ground_truth=Y_test;
accuracy = sum(predictions == ground_truth )/length(ground_truth);
disp(accuracy)



  
