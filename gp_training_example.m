run('toolbox/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc
rng(1)
%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %inference method
num_trs = 1000; %number of training points
num_tes = 200; %number of test points
train_iter = 40;



modification

[X_train,y_train,X_test,y_test] = data_loading(...)



%% training of the GP
disp('Training GP')
meanfunc = @meanZero;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));
