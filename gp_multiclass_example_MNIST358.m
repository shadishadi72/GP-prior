clear all
clc
close all


rng(1)
addpath(genpath('checkGP-classification/'))
addpath(genpath('toolbox/gpstuff-master/'))

num_trs = 500;
num_tes = 200;




[X_train,y_train,X_test,y_test ] = load_mnist358(num_trs,num_tes); %Change this with new dataset



%% GP Training
disp('Training the GP - could take a few minutes...')
% Create covariance functions
gpcf1 = gpcf_sexp('lengthScale', ones(1,size(X_train,2)), 'magnSigma2', 1);
% Set the prior for the parameters of covariance functions
pl = prior_t();%('s2',1,'nu',1);
pm = prior_sqrtt();%('s2',1,'nu',1);
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pm);
% Create the GP structure
gp = gp_set('lik', lik_softmax, 'cf', gpcf1, 'jitterSigma2', 1e-2);
% ------- Laplace approximation --------
fprintf(['Softmax model with Laplace integration over the latent\n' ...
    'values and MAP estimate for the parameters\n'])
% Set the approximate inference method
gp = gp_set(gp, 'latent_method', 'Laplace');
% Set the options for the optimization
opt=optimset('TolFun',0.1,'TolX',1e-3,'Display','iter','MaxIter',10);
% Optimize with the BFGS quasi-Newton method
gp=gp_optim(gp,X_train,y_train,'opt',opt,'optimf',@fminlbfgs);

[Eftep, Covftep, lpytep] = gp_pred(gp, X_train, y_train, X_test,'yt', ones(size(X_test,1),size(y_train,2)));

% calculate the percentage of misclassified points
ttep = (exp(lpytep)==repmat(max(exp(lpytep),[],2),1,size(lpytep,2)));
disp(['The percentage of misclassified points: ' num2str((sum(sum(abs(ttep-y_test)))/2)/size(y_test,1))])