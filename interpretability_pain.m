
clear all
close all
clc
global fs_d



rng(1)


run('toolbox/gpml-matlab-master/startup.m')

addpath(genpath('checkGP-classification'))

load('trained_model_pain')


subjIdx = 1;

X_train = trained_model_pain.X_train;
X_test = trained_model_pain.X_test;
hyp = trained_model_pain.hyp;



fs_d = 32;
likfunc = @likErf;
infFun = @infEP;
meanfunc = @simple_feature_phasic;
covfunc = @covSEard;

[a, b, c, pred_var, lp, post_local] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train(:,[1:end-1]), X_train(:,end), X_test(:,[1:end-1]),ones(size(X_test,1), 1));
pred_label=zeros(size( X_test,1),1);
pred_label(exp(lp) >= 0.5) = 1;  %% rest
pred_label(exp(lp) < 0.5) = -1;  %% cpt

% 
global post
post = post_local;
% 
% %Code parameters
bound_comp_opts = default_parameter_checkGP_OAT('points_to_analyse_vec',size(X_test,1));
 
[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,size(X_train,1),infFun,meanfunc,covfunc,likfunc);

%Check I am getting the correct parameters
%check_manual_and_tool_prediction_agree(X_test(:,1:(end-1)),X_train(:,1:(end-1)),params_for_gp_toolbox,trainedSystem,S,c,pred_var)
% 
global training_data
global training_labels

global loop_vec2
training_data = X_train(:,1:(end-1));
training_labels = X_train(:,end);


S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);


%
pi_LLs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_UUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_LUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
pi_ULs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
feature_magnitude = cell(length(bound_comp_opts.points_to_analyse_vec),1);
%loop over test points
 
 for idx_point = 1:length(bound_comp_opts.points_to_analyse_vec)
    disp(['Test point idx: ' int2str(idx_point) '/' int2str(length(bound_comp_opts.points_to_analyse_vec)) ]);
    testIdx = bound_comp_opts.points_to_analyse_vec(idx_point);
    testPoint = X_test(testIdx,1:(end-1));
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
%  
% 
feature_importance = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
for mm = 1:length(bound_comp_opts.points_to_analyse_vec)

%     feature_importance = feature_importance +
%     feature_magnitude{idx_point};I think idx_point is wrong 
        feature_importance = feature_importance + feature_magnitude{mm};

end
feature_importance = feature_importance/length(bound_comp_opts.points_to_analyse_vec);
% 
% % figure
% % plot(bound_comp_opts.epsilons_vec,feature_importance,'LineWidth',2.0);
% % grid on
% % xlabel('epsilon');
% % ylabel('Average Feature Importance');
% % % leg={'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10'};
% % % legendflex(leg);'
% 
% % %%  see interpret    
% % % figure 
% % % plot(lfnu')
% % signals=feature_importance;
% % time=[1:size(feature_importance,2)];
% % xx=nanmedian(signals);
% % yy=mad(signals,1)*1.42626/sqrt(size(signals,1));
% % delt=1.26*nanmedian(yy);
% % 
% % figure
% % t=time;
% % fillhdl = fill([t,t(end:-1:1)],[xx-yy,fliplr(xx+yy)],'r');
% % set(fillhdl,'facecolor','k','edgecolor','k','facealpha',.2,'edgealpha',.2)
% % hold on; plot(t,xx','color','k','linewidth',2);
% 
% 
% 
%    
%      
%      feature_importance_pain_feat05{ii,1}= feature_importance;
%     
%     
% end
%toc