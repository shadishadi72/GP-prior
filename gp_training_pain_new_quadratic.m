
clear all
clc
% clc
% cd('/Users/shadi/Desktop/PhD_projects/GP_work/GP-prior')
run('toolbox/gpml-matlab-master/startup.m')

addpath(genpath('./'))

load('dataset_pain_04')

dataset_hrveda=dataset_pain_04;


%% just EDA%%

rng(1)
%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %inference method





% downsample
fs=32;
fs_d=32;
rate_down=fs/fs_d;
% downsample data
dataset_hrveda_down=downsample(dataset_hrveda(:,[2:end-1])',rate_down)';
dataset_hrveda_downsamp=zeros(size(dataset_hrveda,1),size(dataset_hrveda_down,2)+2);
dataset_hrveda_downsamp(:,1)=dataset_hrveda(:,1);
dataset_hrveda_downsamp(:,[2:end-1])=dataset_hrveda_down;
dataset_hrveda_downsamp(:,end)=dataset_hrveda(:,end);


dataset_fin=dataset_hrveda_downsamp;


% % for LOSO
subj=dataset_fin(:,1);
dataset_fin=dataset_fin(:,[2:end]);


all_label=[];
all_lpex=[];
all_pred=[];
train_iter = 25;


tic
delete(gcp('nocreate'))
parpool(30)
parfor ii=1:max(subj)
    disp( ii)
    %
    class_score=0;
    
    
    %% for LOSO
    %
    ix=find(subj==ii);
    trainingSet=dataset_fin;
    X_test=trainingSet(ix,:);
    
    trainingSet(ix,:)=[];
    X_train=trainingSet(:,:);
    
    
    %% training of the GP
    disp('Training GP')
    
    ell =2;
    sf =2;
    hyp=struct();
    
    % choice for cov functions
    covfunc = @covSEard;
    hyp.cov = log([ones(1,size(X_train,2)-1)*ell, sf]);
    % %
    
    
    
    
    %  % adding features in priro
    no_feat=8;
    %meanfunc = @simple_feature_phasic;
    
    %hyp.mean = log(ones(no_feat,1));
    
    %hyp for poly
    degree=2;
    meanfunc = {@simple_feature_phasic_poly,degree};
    hyp.mean = log(ones(degree*no_feat,1));
    
    
  
    
    

    
    
    
    hyp2 = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train(:,[1:end-1]),X_train(:,end));
    
    
    [a, b, c, pred_var, lp, post_local] = gp(hyp2, infFun, meanfunc, ...
        covfunc, likfunc, X_train(:,[1:end-1]), X_train(:,end), X_test(:,1:end-1),ones(size(X_test,1), 1));
    %
    pred_label=zeros(size( X_test,1),1);
    pred_label(exp(lp) >= 0.5) = 1; %% rest
    pred_label(exp(lp) < 0.5) = -1;  %% cpt
    
    
    
    label=X_test(:,end);
    all_label=[all_label;label];
    
    lpex=exp(lp);
    all_lpex=[all_lpex;lpex];
    
    pred=pred_label;
    all_pred=[all_pred;pred];
    
    %  %% interpretability
    % %
    X_train_r=X_train(:,[1:end-1]);
    X_test_r=X_test(:,[1:end-1]);
    y_train_r=X_train(:,end);
    trained_model_pain(ii).X_train=X_train;
    trained_model_pain(ii).X_test=X_test;
    trained_model_pain(ii).hyp=hyp2;
    
end

%
% global post
% post = post_local;
%
% %Code parameters
% bound_comp_opts = default_parameter_checkGP_OAT();
%
% [trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp2,size(X_train_r,1),infFun,meanfunc,covfunc,likfunc);
%
% %Check I am getting the correct parameters
% check_manual_and_tool_prediction_agree(X_test_r,X_train_r,params_for_gp_toolbox,trainedSystem,S,c,pred_var)
%
% global training_data
% global training_labels
%
% global loop_vec2
% training_data = X_train_r;
% training_labels = y_train_r;
%
% clear X_train_r y_train_r
% clear Kstar Kstarstar
%
% S = S*params_for_gp_toolbox.sigma;
% global R_inv
% global U
% global Lambda
% R_inv = S;
% R_inv = 0.5*(R_inv + R_inv');
% [U,Lambda] = eig(R_inv);
% Lambda = diag(Lambda);
%
% pi_LLs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% pi_UUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% pi_LUs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% pi_ULs = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% bound_comp_opts.points_to_analyse_vec=size(X_test_r,1); % shadi's change
% feature_magnitude = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% %loop over test points
%
%  for idx_point = 1:length(bound_comp_opts.points_to_analyse_vec)
%     disp(['Test point idx: ' int2str(idx_point) '/' int2str(length(bound_comp_opts.points_to_analyse_vec)) ]);
%     testIdx = bound_comp_opts.points_to_analyse_vec(idx_point);
%     testPoint = X_test_r(testIdx,:);
%     if strcmp(bound_comp_opts.pix_2_mod,'all')
%         bound_comp_opts.pix_2_mod = 1:length(testPoint);
%     end
%     %loop over epsilon values
%     pi_LLs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
%     pi_UUs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
%     pi_LUs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
%     pi_ULs{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
%     feature_magnitude{idx_point} = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
%     for idx_eps = 1:length(bound_comp_opts.epsilons_vec)
%         bound_comp_opts.epsilon = bound_comp_opts.epsilons_vec(idx_eps);
%         disp(['Analysis for epsilon = ', num2str(bound_comp_opts.epsilon) ]);
%         assert(strcmp(bound_comp_opts.mod_modus,'OAT'));
%
%         %loop over features
%         for idx_pixel = 1:length(bound_comp_opts.pix_2_mod)
%             bound_comp_opts.pix_2_mod_curr = bound_comp_opts.pix_2_mod(idx_pixel);
%             [bound_comp_opts.x_L, bound_comp_opts.x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
%                 bound_comp_opts.pix_2_mod_curr,bound_comp_opts.constrain_2_one);
%             [pi_LLs{idx_point}(idx_eps,idx_pixel),pi_UUs{idx_point}(idx_eps,idx_pixel), pi_LUs{idx_point}(idx_eps,idx_pixel),...
%                 pi_ULs{idx_point}(idx_eps,idx_pixel),count,exitFlag] = ...
%                 main_pi_hat_computation('all',testPoint,testIdx,...
%                 params_for_gp_toolbox,bound_comp_opts,trainedSystem,S);
%             feature_magnitude{idx_point}(idx_eps,idx_pixel) = pi_UUs{idx_point}(idx_eps,idx_pixel) - pi_LLs{idx_point}(idx_eps,idx_pixel);
%
%         end
%
%     end
%
% end
%
%
% feature_importance = zeros(length(bound_comp_opts.epsilons_vec),length(bound_comp_opts.pix_2_mod));
% for mm = 1:length(bound_comp_opts.points_to_analyse_vec)
%
% %     feature_importance = feature_importance +
% %     feature_magnitude{idx_point};I think idx_point is wrong
%         feature_importance = feature_importance + feature_magnitude{mm};
%
% end
%  feature_importance = feature_importance/length(bound_comp_opts.points_to_analyse_vec);
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
toc
%% select the max among all_pred for all wind
%   all_pred_win{mm,1}=all_pred;

%   end

% %  majority voting
%
%  all_pred_win_mat=cell2mat(all_pred_win');
%  Final_decision = zeros(size(all_pred_win,1),1);

% for i=1:size(all_pred_win_mat,1)
% class1=length(find(all_pred_win_mat(i,:)==1));
% class2=length(find(all_pred_win_mat(i,:)==-1));
% tosee(i,1)=class1;
% tosee(i,2)=class2;



% if class1>class2
%     Final_decision(i,1)=1;
% else
%     Final_decision(i,1)=-1;
% end
% end
%
%
%
%
%
%  performance

%    for i=1:size(all_pred_win_mat,2)
%        Final_decision=all_pred_win_mat(:,i);
Final_decision=all_pred;
acc = sum(Final_decision == all_label(:,end))/size(all_label,1)*100;
cd=confusionmat(Final_decision,all_label(:,end));




se=cd(1,1)/(cd(1,1)+cd(2,1))*100;
sp=cd(2,2)/(cd(1,2)+cd(2,2))*100;

% when the inference fails
indi=find(Final_decision);
deci2=Final_decision(indi);
label2=all_label(indi);
cd2=confusionmat(deci2,label2);
se2=cd2(1,1)/(0.5*size(dataset_fin,1))*100;
sp2=cd2(2,2)/(0.5*size(dataset_fin,1))*100;
acc2=(se2+sp2)/2;


res=[se;sp;acc];
%    res_all(i).res=100-res;
%    end
disp(res)

% for F1 score
%% calcul de la précision , le rappel (recall) et le F-score
%% recall
confMat=cd;
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
end
recall(isnan(recall))=[];

Recall=sum(recall)/size(confMat,1);
%% précision
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
Precision=sum(precision)/size(confMat,1);
%% F-score
F_score=2*Recall*Precision/(Precision+Recall); %%F_score=2*1/((1/Precision)+(1/Recall));
disp(F_score)

