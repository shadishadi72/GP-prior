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
% [X_train,X_test] = normalise_train_test(X_train,X_test);


%% training of the GP
disp('Training GP')
%meanfunc = @meanLinear;
meanfunc = @meanLinear;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
hyp.mean = log(ones(size(X_train,2),1));
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post_local] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));


pred_labels = zeros(size(y_test));
pred_labels(exp(lp) >= 0.5) = 1;
pred_labels(exp(lp) < 0.5) = -1;
acc = sum(pred_labels == y_test)/length(y_test);
disp(acc)
cd=confusionmat(pred_labels,y_test)
se=cd(1,1)/(size(X_train,1)/2);
sp=cd(2,2)/(size(X_train,1)/2);
res=[se,sp,acc]

%%%%%%%%%%%
%Check-GP%%
%%%%%%%%%%%
global post
post = post_local;

%Code parameters
bound_comp_opts = default_parameter_checkGP_OAT();

[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,size(X_train,1),infFun,meanfunc,covfunc,likfunc);

%Check I am getting the correct parameters
check_manual_and_tool_prediction_agree(X_test,X_train,params_for_gp_toolbox,trainedSystem,S,c,pred_var)

global training_data
global training_labels

global loop_vec2
training_data = X_train;
training_labels = y_train;

clear X_train y_train
clear Kstar Kstarstar

S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

pi_LLs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_UUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_LUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_ULs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
feature_magnitude = cell(length(bound_comp_opts.points_to_analyse_vec),1);
%loop over test points
for idx_point = 1:length(bound_comp_opts.points_to_analyse_vec)
    disp(['Test point idx: ' int2str(idx_point) '/' int2str(length(bound_comp_opts.points_to_analyse_vec)) ]);
    testIdx = bound_comp_opts.points_to_analyse_vec(idx_point);
    testPoint = X_test(testIdx,:);
    if strcmp(bound_comp_opts.pix_2_mod,'all')
        bound_comp_opts.pix_2_mod = 1:length(testPoint);
    end
    %loop over epsilon values
    pi_LLs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
    pi_UUs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
    pi_LUs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
    pi_ULs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
    feature_magnitude{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
    for idx_eps = 1:length(bound_comp_opts.epsilons_vec)
        bound_comp_opts.epsilon = bound_comp_opts.epsilons_vec(idx_eps);
        disp(['Analysis for epsilon = ', num2str(bound_comp_opts.epsilon) ]);
        assert(strcmp(bound_comp_opts.mod_modus,'OAT'));
        
        %loop over features
        for idx_pixel = 1:length(bound_comp_opts.pix_2_mod)
            bound_comp_opts.pix_2_mod_curr = bound_comp_opts.pix_2_mod(idx_pixel);
            [bound_comp_opts.x_L, bound_comp_opts.x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
                bound_comp_opts.pix_2_mod_curr,bound_comp_opts.constrain_2_one);
            [pi_LLs{idx_point}(idx_eps,idx_pixel),pi_UUs{idx_point}(idx_eps,idx_pixel), pi_LUs{idx_point}(idx_eps,idx_pixel),...
                pi_ULs{idx_point}(idx_eps,idx_pixel),count,exitFlag] = ...
                main_pi_hat_computation('all',testPoint,testIdx,...
                params_for_gp_toolbox,bound_comp_opts,trainedSystem,S);
            feature_magnitude{idx_point}(idx_eps,idx_pixel) = pi_UUs{idx_point}(idx_eps,idx_pixel) - pi_LLs{idx_point}(idx_eps,idx_pixel);
            
        end
        
    end
    
end


feature_importance = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
for ii = 1:length(bound_comp_opts.points_to_analyse_vec)
    feature_importance = feature_importance + feature_magnitude{idx_point};
end
feature_importance = feature_importance/length(bound_comp_opts.points_to_analyse_vec);


plot(bound_comp_opts.epsilons_vec,feature_importance,'LineWidth',2.0)
grid on
xlabel('epsilon');
ylabel('Average Feature Importance');
legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7'})

