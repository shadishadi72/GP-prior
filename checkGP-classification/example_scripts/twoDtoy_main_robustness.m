%Script for Robustness analysis of sysnthetic 2-d dataset (Results shown in Figure 3).

run('toolboxes/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc

addpath(genpath('utils/'))

%Code parameters
savingFlag = true;
numberOfThreads = 1;


%GP model parameters
likfunc = @likErf; %link function
infFun = @infLaplace; %inference method
num_trs = 1000; %number of training points
num_tes = 200; %number of test points
train_iter = 40;

%Branch and bound parameters
maxiters = 15000; 
tollerance = 0.01;
epsi = 0.5;
Testpoints = [30]; 
bound_comp_opts.mod_modus = 'ls';
bound_comp_opts.pix_2_mod = [];
bound_comp_opts.constrain_2_one = false;
bound_comp_opts.max_iterations = maxiters;
bound_comp_opts.tollerance = tollerance;
bound_comp_opts.N = 100;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-20;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.likmode = 'analytical';
bound_comp_opts.mode = 'binarypi';

%%%

rng(1)
maxNumCompThreads(numberOfThreads);
dirName = strcat('results/',datestr(datetime('now')),'Robustness_2Dtoy');
if exist(dirName,'dir') ~=7
    if savingFlag
        mkdir(dirName)
    end
end


%some global variables used for time monitoring of the method
global mu_time
global std_time
global discrete_time
global inference_time
discrete_time = 0;
mu_time = 0;
std_time = 0;
inference_time = 0;
sample_time = 0;
bound_time = 0;
bound_time2 = 0;
lime_time = 0;
%global variables for posteriori GP
global pred_var
global post


%Load data
[X_train,y_train,X_test,y_test] = generate_2d_synthetic_datasets(num_trs,num_tes);


%% training of the GP
disp('Training GP')
meanfunc = @meanZero;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
%
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[~, ~, ~, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));



[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,num_trs,infFun,meanfunc,covfunc,likfunc);

disp('Done with training')
%%





if bound_comp_opts.numberOfThreads > 1
    if isempty(gcp('nocreate'))
        parpool(bound_comp_opts.numberOfThreads);
    end
end





%% Predefine outputs
epsilons = [0,epsi];
n_eps = length(epsilons);
BoundResult.analytical = zeros(1+length(Testpoints),n_eps+1);
BoundResult.analytical(2:end,1) = exp(lp(Testpoints));
BoundResult.analytical(1,1) = epsi;
BoundResult.analytical_exact = zeros(1+length(Testpoints),n_eps+1);
BoundResult.analytical_exact(2:end,1) = exp(lp(Testpoints));
BoundResult.analytical_exact(1,1) = epsi;
iter_count.Laplace = zeros(1+length(Testpoints),n_eps+1);
iter_count.Laplace(1,1) = epsi;
flags.Laplace = zeros(1+length(Testpoints),n_eps+1);
flags.Laplace(1,1) = epsi;


%Storing stuff as global variables for time saving reasons
global training_data
global training_labels
global loop_vec2
training_data = X_train;
training_labels = y_train;
loop_vec2 = discretise_real_line(bound_comp_opts.N);

clear Kstar Kstarstar data latent_variance_prediction

S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

%% For loop over test points
for dd = 1:length(Testpoints)
    
    testIdx = Testpoints(dd);
    disp('Current test point:')
    disp(testIdx)
    testIdx = Testpoints(dd);
    testPoint = X_test(testIdx,:);
    [~,bound_comp_opts.pix_2_mod] = maxk(params_for_gp_toolbox.theta_vec,2);
    
    for ii = 2:n_eps
        bound_comp_opts.epsilon = epsilons(ii);
        disp('Current epsilon:')
        disp(bound_comp_opts.epsilon)
        
        [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
            bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
        bound_comp_opts.x_L = x_L;
        bound_comp_opts.x_U = x_U;
        
        
        iter_count.Laplace(dd+1,1) = testIdx;
        flags.Laplace(dd+1,1) = testIdx;
        aux = tic;
        
        [pi_LL,pi_UU, pi_LU,pi_UL,count,exitFlag] = main_pi_hat_computation('all',testPoint,testIdx,...
            params_for_gp_toolbox,bound_comp_opts,trainedSystem,S);
        
        BoundResult.analytical(dd+1,ii) = pi_LL;
        BoundResult.analytical_exact(dd+1,ii) = pi_LU;
        iter_count.Laplace(dd+1,ii) = count.min;
        flags.Laplace(dd+1,ii) = exitFlag.min;
        
        BoundResult.analytical(dd+1,ii+1) = pi_UU;
        BoundResult.analytical_exact(dd+1,ii+1) = pi_UL;
        iter_count.Laplace(dd+1,ii+1) = count.max;
        flags.Laplace(dd+1,ii+1) = exitFlag.max;
        
        toc(aux)
        
        
        
    end
    
    
    
    disp(strcat('DONE WITH TESTPOINT ', num2str(testIdx)))
    
end


%% Save parameters and settings used

saveables.epsilon = epsi;
saveables.training_iters = train_iter;
%saveables.modelAcc = acc;
saveables.boundOpts = bound_comp_opts;
saveables.covfunc = covfunc;
saveables.infFun = infFun;
saveables.likfunc = likfunc;
saveables.meanfunc = meanfunc;
saveables.num_tes = num_tes;
saveables.num_trs = num_trs;
saveables.params_for_gp_toolbox = params_for_gp_toolbox;
saveables.Testpoints = Testpoints;
saveables.BoundResult = BoundResult;
saveables.iter_count = iter_count;
saveables.exitFlags = flags;
saveables.BoundResult_diff = BoundResult.analytical(2:end,3) - BoundResult.analytical(2:end,2);

if savingFlag
    save(strcat(dirName,'/ParamsAndSettings.mat'),'saveables');
    
    disp('Parameters and settings succesfully saved')
end

