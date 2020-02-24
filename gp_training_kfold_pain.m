run('toolbox/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc

addpath(genpath('checkGP-classification/'))

 load('dataset_pain_04.mat')
 
 
% madalena {'mean_t', 'std_t', 'mean_r', 'std_r', 'n_peak', 'max_peak', 'sum_amp''};
%% just EDA%% 

rng(1)
%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %inference method

train_iter = 20;

dataset_hrveda=dataset_pain_04;
 
%   dataset_hrveda=dataset_pain_04([1:5,5+87:19+87],[1:5,end]);


 % selecting samples
%  dataset_hrveda(:,[2:30,100:129])=[];
 
%% extracting features 
% fs_d=32;
% for i=1:size(dataset_pain_04,1)
%     aa=dataset_pain_04(i,[2:end-1]);
%     aa_feat=run_eda_phasic(aa,fs_d);
%     feat_pain_all(i,:)=aa_feat;
% end
% 
%  dataset_hrveda_feat=zeros(size(dataset_pain_04,1),size(aa_feat,2)+2);
%  dataset_hrveda_feat(:,1)=dataset_pain_04(:,1);
%   dataset_hrveda_feat(:,end)=dataset_pain_04(:,end);
%     dataset_hrveda_feat(:,[2:end-1])= feat_pain_all;
% % 
% % 
%    dataset_hrveda= dataset_hrveda_feat;

%     




subj=dataset_hrveda(:,1);
dataset_hrveda(:,1)=[];
% selecting the last  part 
% dataset_hrveda(:,[1:120])=[];


% some random data 
% dataset_hrveda2=zeros(174,130);
% dataset_hrveda2(1:174,1:129)=randn(174,129);
% dataset_hrveda2(1:174,130)=dataset_hrveda(:,130);
% dataset_hrveda=dataset_hrveda2;

Acc_feat_sel=[];

% % for LOSO
% All_predicted=zeros(size(dataset_hrveda,1)/2,1);
% All_labels=zeros(size(dataset_hrveda,1)/2,1);

%% for Kfold 
fold=10;
indices = crossvalind('Kfold',dataset_hrveda(:,end),fold);
all_label=[];
all_lpex=[];
all_pred=[];

%  for ii =1:fold
%     ii
      % LOSO
      tic
  for ii=1:max(subj)
    disp( ii)
     
     class_score=0;
%      for kfold
%          test = (indices == ii);
%          ix=find(test);
%          train = ~test;
%           X_train=dataset_hrveda(train,:);
%          X_test=dataset_hrveda(test,:);
     
     %% for LOSO
% 
      ix=find(subj==ii);
        trainingSet=dataset_hrveda;
        X_test=trainingSet(ix,:);

        trainingSet(ix,:)=[];
        X_train=trainingSet(:,:);
    
    
    
    % Zscore normalization%
%          [X_train(:,1:end-1),mu,sigma] = my_zscore2(X_train(:,1:end-1));
    % max-min normalization%
%       [X_train(:,1:end-1)] = normalise(X_train(:,1:end-1));
    
    
    %% training of the GP
    disp('Training GP')
    
    covfunc = @covSEard;
    ell = 0.5;
    sf =0.5;
    hyp.cov = log([ones(1,size(X_train,2)-1)*ell, sf]);
    
    %%mean functions
%          meanfunc = @meanZero;
    
    %  meanfunc = @meanConst;
    
%         meanfunc = @meanLinear;
%       hyp.mean = log(ones(size(X_train,2)-1,1));
      
% ml = {@meanLinear};
% % meanfunc=@meanMask;
% mask = [false,true,false,true,true,true,true]; % mask excluding all but the 2nd component
% meawnfunc = {'meanMask',mask,ml}; 
% hyp.mean =  hyp.mean(mask);





%      meanfunc = {@linear_excluding,[2,4:7]};
%         no_feat=2;
%     hyp.mean = log(ones(size(X_train,2)-1,1));
   

%  % adding features in priro 
  meanfunc = @simple_feature_phasic;
  no_feat=8;
 hyp.mean = log(ones(no_feat,1));   
    
    
    
    % % hyp for mean constant
    % % hyp.mean =mean(X_train(:,1))+mean(X_train(:,2))+mean(X_train(:,3))+mean(X_train(:,4))+mean(X_train(:,5));
    % % hyp.mean =1;
    %
    % % hyp for mean linear
    
    %%hyp for poly
    % degree=2;
    % meanfunc = {@meanPoly,degree};
    % hyp.mean = log(ones(degree*size(X_train,2),1));
    
    %%hyp for meanWSPC
    % degree=1;
    % meanfunc = {@meanWSPC,degree};
    % hyp.mean = log(ones(degree*size(X_train,2)+2*degree,1));
    
    %%hyp for composite
    % m1=@meanLinear;
    % hypc = log(ones(size(X_train,2),1));
    
    % % meanfunc = {'meanScale',{m1}};
    % hyp.mean = [3;hypc]; % scale by 3
    
    % meanfunc = {'meanPow',2,{m1}};
    % hyp.mean = hypc; % scale by 3
    
    
    % %%hyp for composite (sum )
    % m1=@meanLinear;
    % hyp1 = log(ones(size(X_train,2),1));
    % degree=2;
    % m2 = {@meanWSPC,degree};
    % hyp2 = log(ones(degree*size(X_train,2)+2*degree,1));
    % meanfunc= {'meanSum',{m1,m2}};
    % hyp.mean = [hyp1; hyp2]; % sum
    
    
    
    %%  hyperpriros
    % par = {meanfunc,covfunc,likfunc,X_train,y_train};
    % mfun = @minimize; % input for GP function
    % % a) plain marginal likelihood optimisation (maximum likelihood)
    % % im = @infEP;                                  % inference method
    %
    % %b) regularised optimisation (maximum a posteriori) with 1d priorsprior.mean = {pg;pc}; % Gaussian prior for first, clamp second par
    % p1 = {@priorSmoothBox1 ,0,3,15};
    % prior.cov = {p1;[]}; % box prior for first, nothing for second par
    % im = {@infPrior,@infExact,prior};
    %
    % hyp2 = feval(mfun, hyp, @gp, -train_iter, im, par{:});      % optimise
    
    
    
    hyp2 = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train(:,[1:end-1]),X_train(:,end));
    
    % training
    % [NLZ,DNLZ,posttrain] = gp (hyp, infFun, meanfunc,  covfunc, likfunc, X_train, y_train);
    
    % reusing posterior in prediction
    % [a, b, c, pred_var, lp, post_local]=gp(hyp, infFun, meanfunc, covfunc, likfunc, X_train, posttrain, X_test,ones(size(X_test,1), 1));
    %  NLZ      returned value of the negative log marginal likelihood (like a
    %  cost function)
    %  DNLZ     struct of column vectors of partial derivatives of the negative log marginal likelihood w.r.t. mean/cov/lik hyperparameters
    % prediction
    
    % compare gp training for hyp (just prior) and hyp2(after ptimization
    % NLML)]]
    
    
    [a, b, c, pred_var, lp, post_local] = gp(hyp2, infFun, meanfunc, ...
        covfunc, likfunc, X_train(:,[1:end-1]), X_train(:,end), X_test(:,[1:end-1]),ones(size(X_test,1), 1));
    pred_label=zeros(size( X_test,1),1);
    pred_label(exp(lp) >= 0.5) = 1; %% rest
    pred_label(exp(lp) < 0.5) = -1;  %% cpt
    
% %% for Kfold    
% label=X_test(:,end);
% all_label=[all_label;label];
% 
% lpex=exp(lp);
% all_lpex=[all_lpex;lpex];
% 
% pred=pred_label;
% all_pred=[all_pred;pred];

% %% for LOSO
for k=1:length(ix)
    if(pred_label(k) - X_test(k,end) ==0)
        class_score=class_score+1;
    end
end
All_lp((ii-1)*length(ix)+1:ii*length(ix),1)=exp(lp);
All_predicted((ii-1)*length(ix)+1:ii*length(ix),1)=pred_label;
All_labels((ii-1)*length(ix)+1:ii*length(ix),1)=X_test(:,end);
% 


    
  end
 
 toc
%% For Kfold 


% acc = sum(all_pred == all_label(:,end))/size(all_label,1)*100;
% cd=confusionmat(all_pred,all_label(:,end));
% se=cd(1,1)/(size(all_label,1)/2)*100;
% sp=cd(2,2)/(size(all_label,1)/2)*100;
% res=[se;sp;acc];
% disp(res)
%% for lOSO 
% All_predicted=all2_pred;
acc = sum(All_predicted == All_labels(:,end))/size(All_labels,1)*100;
cd=confusionmat(All_predicted,All_labels(:,end));
se=cd(1,1)/(size(All_labels,1)/2)*100;
sp=cd(2,2)/(size(All_labels,1)/2)*100;
res=[se;sp;acc];

 disp(res)
 
 % to save 
%  writedir='/Users/shadi/Desktop/PhD_projects/GP_work/GP-prior/saved GP models/pain GP saved';
%  gp_model_lin_pcf_tonic.pred=All_predicted;
%  gp_model_lin_pcf_tonic.hyp=hyp2;
% save gp_model_zero_pcf_tonic gp_model_lin_pcf_tonic  
% 
% % %% Robustness 
% % 
% X_train_r=X_train(:,[1:end-1]);
% X_test_r=X_test(:,[1:end-1]);
% y_train_r=X_train(:,end);
% 
% %%%%%%%%%%
% % Check-GP%%
% %%%%%%%%%%
% global post
% post = post_local;
% 
% % Code parameters
% bound_comp_opts = default_parameter_checkGP_OAT();
% 
% [trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp2,size( X_train_r,1),infFun,meanfunc,covfunc,likfunc);
% 
% % Check I am getting the correct parameters
% check_manual_and_tool_prediction_agree( X_test_r, X_train_r,params_for_gp_toolbox,trainedSystem,S,c,pred_var)
% 
% global training_data
% global training_labels
% 
% global loop_vec2
% training_data =  X_train_r;
% training_labels =  y_train_r;
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
% feature_magnitude = cell(length(bound_comp_opts.points_to_analyse_vec),1);
% % loop over test points
% % shadi's change : 
% % for idx_point = 1:size(X_test,1)
% 
% for idx_point = 1:length(bound_comp_opts.points_to_analyse_vec)
% 
%     disp(['Test point idx: ' int2str(idx_point) '/' int2str(length(bound_comp_opts.points_to_analyse_vec)) ]);
%     testIdx = bound_comp_opts.points_to_analyse_vec(idx_point);
%     testPoint =X_test_r(testIdx,:);
%     if strcmp(bound_comp_opts.pix_2_mod,'all')
%         bound_comp_opts.pix_2_mod = 1:length(testPoint);
%     end
% %     loop over epsilon values
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
% %         loop over features
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
% for ii = 1:length(bound_comp_opts.points_to_analyse_vec)
%     feature_importance = feature_importance + feature_magnitude{idx_point};
% end
% feature_importance = feature_importance/length(bound_comp_opts.points_to_analyse_vec);
% % 
% % % plot(bound_comp_opts.epsilons_vec,feature_importance,'LineWidth',2.0)
% % % grid on
% % % xlabel('epsilon');
% % % ylabel('Average Feature Importance');
% % % legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7'})
% % % feature_order_eda_cvx={'SMNA(max)';'SMNA(sum)';'SMNA(nom)';'phasic(mean)';'phasic(std)';'tonic(mean)';'tonic(std)';'tonic(max)';'edasymp'};
% % % feature_order_eda_hrv={'mu','var','lf','hf','lfhf','edasymp','edahf'};
% % % feature_order_hrv={'mu','var','lf','hf','lfhf'};
% % % 
% % % leg={'mu','var','lf','hf','lfhf','edasymp','edahf','SMNA(max)','SMNA(sum)','SMNA(nom)','phasic(mean)','phasic(std)','tonic(mean)','tonic(std)','tonic(max)'};
% % % legendflex(leg);
% % % legend({'Feature 1';'Feature 2';'Feature 3';'Feature 4';'Feature 5';'Feature 6';'Feature 7'})
% % 
% 
% %% plot interper
% 
% % figure 
% % plot(lfnu')
%  leg={'meanPh';'stdPh';'maxPks';'sumPks';'noPks'};
% 
% signals=feature_importance;
% t=[1:size(feature_importance,2)];
% xx=nanmedian(signals);
% yy=mad(signals,1);%*1.42626/sqrt(size(signals,1));
% delt=1.26*nanmedian(yy);
% 
% figure
% subplot(3,1,1)
% fillhdl = fill([t,t(end:-1:1)],[xx-yy,fliplr(xx+yy)],'r');
% set(fillhdl,'facecolor','k','edgecolor','k','facealpha',.2,'edgealpha',.2)
% set(fillhdl,'facecolor','k','edgecolor','w','facealpha',.2)
% hold on; plot(t,xx','color','k','linewidth',2);
% xlabel('Feature index','FontSize',26);
% ylabel('Feature importance','FontSize',26);
% title('meanPh,stdPh,maxPks,sumPks,noPks','FontSize',26)
% 
% subplot(3,1,2)
% data=abs(hyp2.mean);
% plot(data)
% xlabel('Feature index','FontSize',26);
% ylabel('Hyper Parameter (Mean)','FontSize',20);
% subplot(3,1,3)
% data=hyp2.cov;
% plot(data)
% xlabel('[Feature index,sf]','FontSize',26);
% ylabel('Hyper Parameter (Covariance)','FontSize',20);



