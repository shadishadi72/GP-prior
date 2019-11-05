run('toolbox/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc

addpath(genpath('checkGP-classification/'))

load('all_feat_gp.mat')
rng(1)
%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %inference method

train_iter = 10;

[X_train,y_train,X_test,y_test] = data_loading_cpt(all_feat_gp);
[X_train,X_test] = normalise_train_test(X_train,X_test);


%% training of the GP
disp('Training GP')
meanfunc = @meanLinear;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
hyp.mean = log(ones(size(X_train,2),1));
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));


pred_labels = zeros(size(y_test));
pred_labels(exp(lp) >= 0.5) = 1;
pred_labels(exp(lp) < 0.5) = -1;
acc = sum(pred_labels == y_test)/length(y_test);
disp(acc)


%%%%%%%%%%%
%Check-GP%%
%%%%%%%%%%%


%Code parameters
bound_comp_opts = default_parameter_checkGP_OAT();






