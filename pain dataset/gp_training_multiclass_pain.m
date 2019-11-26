clear all
clc
close all


rng(1)
addpath(genpath('checkGP-classification/'))
addpath(genpath('toolbox/gpstuff-master/'))
addpath('/Users/shadi/Desktop/GP-prior/')
addpath('/Users/shadi/Desktop/GP-prior/pain GP')
addpath('/Users/shadi/Desktop/GP-prior/pain GP/pain dataset')

load('/Users/shadi/Desktop/GP-prior/pain GP/pain dataset/mean_parameters_dw5.mat')
load('mean_parameters_w10.mat')
load('mean_parameters_w5.mat')






% [X_train,y_train,X_test,y_test ] = load_mnist358(num_trs,num_tes); %Change this with new dataset
[X_train,y_train,X_test,y_test] = data_loading_pain_mc(mean_parameters_w10);


num_trs = size(X_train,1);
num_tes = size(X_test,1);
no_class=5;


%% GP Training
disp('Training the GP - could take a few minutes...')
% Create covariance functions
gpcf1 = gpcf_sexp('lengthScale', ones(1,size(X_train,2)), 'magnSigma2', 1);
% Set the prior for the parameters of covariance functions
pl = prior_t('mu',2,'s2',1,'nu',1);
pm = prior_sqrtt('mu',2,'s2',1,'nu',1);
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pm);
% Create the GP structure
gp = gp_set('lik', lik_softmax, 'cf', gpcf1, 'jitterSigma2', 1e-2);
% ------- Laplace approximation --------
fprintf(['Softmax model with Laplace integration over the latent\n' ...
    'values and MAP estimate for the parameters\n'])
% Set the approximate inference method
gp = gp_set(gp, 'latent_method', 'Laplace');
% Set the options for the optimization
opt=optimset('TolFun',0.01,'TolX',1e-3,'Display','iter','MaxIter',50);
% Optimize with the BFGS quasi-Newton method
gp=gp_optim(gp,X_train,y_train,'opt',opt,'optimf',@fminlbfgs);

[Eftep, Covftep, lpytep] = gp_pred(gp, X_train, y_train, X_test,'yt', ones(size(X_test,1),size(y_train,2)));

pred_prob = reshape(exp(lpytep),num_tes,no_class);
[~, predictions ] = max(pred_prob,[],2);
[~, ground_truth ] = max(y_test,[],2);

accuracy = sum(predictions == ground_truth )/length(ground_truth);
disp(accuracy)