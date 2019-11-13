run('toolbox/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc
addpath('/Users/shadi/Desktop/GP-prior')
addpath('/Users/shadi/Desktop/GP-prior/cpt_cvx_features')
rng(1)
%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %infLaplace %inference method

% eda as input 
load('eda_cp_rest_red_mat.mat')
load('eda_cp_red_mat.mat')
%  for i=1:100
i=15;
train_iter = i;
% single point 
%  [X_train,y_train,X_test,y_test] = data_loading_cpt(all_feat_gp);
% [X_train_f,y_train_f,X_test_f,y_test_f] = data_loading_cpt_cvx(feat_eda_cold,feat_eda_cold_rest,cold_new_zscore,cold_rest_new_zscore);
[X_train_sig,y_train_sig,X_test_sig,y_test_sig] = data_loading_cpt_sig(eda_cp_red_mat,eda_cp_rest_red_mat);



X_train=[X_train_sig];
X_test=[X_test_sig];
y_train=y_train_sig;
y_test=y_test_sig;


 
% % 
% X_train=X_train(:,[9]);
% X_test=X_test(:,[9]);

 
% % time series 
% [X_tr,y_t,X_te,y_te] = data_loading_cpt(all_feat_gp);
% 
% training for all samples 
% no_samp=size(X_train{1,1},2);
% X_train=zeros(no_sub,no_feat);
% 
% for i=1:no_samp
%     for j=1:no_feat
% b= cell2mat(X_tr(:,no_feat));
% X_train(:,j)=b(:,i);
%     end
% end
    
    
%% training of the GP
disp('Training GP')




covfunc = @covSEard;
ell = 2;
sf = 2;

%meanfunc = @(varargin)(linear_excluding([5,7],varargin{:}));
% meanfunc = {@linear_excluding,[1:size(X_train_sig,2)]};
meanfunc = @simple_feature;
no_feat=8;

%  meanfunc = @meanZero;

% hyp.mean = log(ones(size(X_train,2),1));
% hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);


%hyp.mean = log(ones(size(X_train,2),1));
hyp.mean = log(ones(no_feat,1));


hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);

%hyp.mean = [1;1];
degree = 1;
%meanfunc = {@meanPoly,degree};
%hyp.mean = log(ones(degree*size(X_train,2),1));


hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));

%% accuracy 
pred_labels = zeros(size(y_test));
pred_labels(exp(lp) >= 0.5) = 1;
pred_labels(exp(lp) < 0.5) = -1;
acc(1) = sum(pred_labels == y_test)/length(y_test);
% end
i
disp(acc)
