function [trainedSystem,S,params_for_gp_toolbox]= build_gp_trained_params(hyp,n_tr,infFun,meanfunc,covfunc,likfunc)

global post

%getting trained System out of the trained gp:
% this is the vector such that latent_mu = trainedSystem*Covariance(test)
trainedSystem = post.alpha;


%Checking Variance here
S = - post.L(eye(n_tr,n_tr));


params_for_gp_toolbox.hyp = hyp;
params_for_gp_toolbox.infFun = infFun;
params_for_gp_toolbox.meanfunc = meanfunc;
params_for_gp_toolbox.covfunc = covfunc;
params_for_gp_toolbox.likfunc = likfunc;
params_for_gp_toolbox.sigma =  exp(hyp.cov(end)).^2;
params_for_gp_toolbox.theta_vec = 1./(2.*(exp(hyp.cov(1:(end-1))).^2));


end