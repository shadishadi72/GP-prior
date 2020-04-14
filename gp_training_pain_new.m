
clear all
clc
% clc
% cd('/Users/shadi/Desktop/PhD_projects/GP_work/GP-prior')

run('toolbox/gpml-matlab-master/startup.m')

addpath(genpath('/Users/shadi/Desktop/PhD_projects/GP_work/GP-prior'))

load('dataset_pain_04')

dataset_hrveda=dataset_pain_04;

%   dataset_hrveda([100:103,204:208],[2:end-1])=normalise(dataset_hrveda([100:103,204:208],[2:end-1]));

% dataset_hrveda=dataset_hrveda([1:198],:);
%  dataset_hrveda(:,[2:end-1])=zscore(dataset_hrveda(:,[2:end-1]));

% dataset_hrveda=dataset_hrveda([1:99,104:203],:);

% madalena {'mean_t', 'std_t', 'mean_r', 'std_r', 'n_peak', 'max_peak', 'sum_amp''};
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




% preprocessing data with having tonic+phasic
%  dataset_for_feat=get_tonic(dataset_hrveda_downsamp,fs_d);

%   dataset_for_feat=dataset_hrveda_downsamp;

% %% doing windowing in the input (data samples are large)
% % comment if you wanna do features
%
% win=fs_d*5;    % windows of 10 s
% overlap=fs_d*4;
% lap=0.2;  % (5-4)/5
% 
% nadvance = win - overlap;
% nrecord  = fix ( (size(dataset_for_feat,2)-2- overlap) / nadvance );
% 
%  for kk=1:nrecord
% %
%     if kk==1
%  data_win{kk,1}=dataset_for_feat(:,[2:win+1]);
% 
%     else
%  data_win{kk,1}=dataset_for_feat(:,[ floor((kk-1)*lap*win): floor((kk-1)*lap*win+win-1)]);
%     end
%  end

 % train GP for each window

%  all_pred_win=cell(nrecord,1);

%  for mm=1%:nrecord

%  disp(mm)
% inp_gp=zeros(size(data_win{mm,1},1),size(data_win{mm,1},2)+2);
% inp_gp(:,1)=dataset_hrveda(:,1);
% inp_gp(:,[2:end-1])=data_win{mm,1};
% inp_gp(:,end)=dataset_hrveda(:,end);

% dataset_fin=inp_gp;


%  extracting features for EDA
%  dataset_for_feat=dataset_hrveda_downsamp;
% for i=1:size(dataset_for_feat,1)
%     aa=dataset_for_feat(i,[2:end-1]);
%     aa_feat=run_eda_phasic(aa,fs_d);
%     feat_pain_all(i,:)=aa_feat;
% end
% 
% dataset_hrveda_feat=zeros(size(dataset_for_feat,1),size(aa_feat,2)+2);
% dataset_hrveda_feat(:,1)=dataset_for_feat(:,1);
% dataset_hrveda_feat(:,end)=dataset_for_feat(:,end);
% dataset_hrveda_feat(:,[2:end-1])= feat_pain_all;
% % % 
% % % 
%       dataset_fin=dataset_hrveda_feat;

   
  dataset_fin=dataset_hrveda_downsamp;

% dataset_fin([1:50],end)=-1;
% 
% dataset_fin([150:end],end)=1;

% % for LOSO
subj=dataset_fin(:,1);
dataset_fin=dataset_fin(:,[2:end]);
%% for Kfold
% fold=10;
% indices = crossvalind('Kfold',dataset_hrveda(:,end),fold);


all_label=[];
all_lpex=[];
all_pred=[];
train_iter =25;


%  for ii =1:fold
%     ii
%       LOSO
tic
delete(gcp('nocreate'))
parpool(15)
parfor ii=1:max(subj)
    disp( ii)
    %
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
    trainingSet=dataset_fin;
    X_test=trainingSet(ix,:);
    
    trainingSet(ix,:)=[];
    X_train=trainingSet(:,:);
    
    %     [X_train(:,1:end-1),mu,sigma]=my_zscore2(trainingSet(:,1:end-1));
    
    
    
    
    % Zscore normalization%
%             [X_train(:,1:end-1),mu,sigma] = my_zscore(X_train(:,1:end-1));
    % max-min normalization%
      %        [X_train(:,1:end-1)] = normalise(X_train(:,1:end-1));
    
    
    %% training of the GP
    disp('Training GP')
    
    ell =2;
    sf =2;
    hyp=struct();
    
    % choice for cov functions
    covfunc = @covSEard;
      hyp.cov = log([ones(1,size(X_train,2)-1)*ell, sf]);
% %     
    
    %
%     
%            covfunc = @covSEiso;
%              hyp.cov = log([ell;sf]); % isotropic Gaussian
% %     %
    
    %%mean functions
                   %      meanfunc = @meanZero;
%    
%        meanfunc = @meanConst;
    %
%     meanfunc = @meanLinear;
%   hyp.mean = log(ones(size(X_train,2)-1,1));
    %
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
%       no_feat=1;
         hyp.mean = log(ones(no_feat,1));
    
    
    
%     hyp for mean constant
%     hyp.mean =mean(X_train(:,1))+mean(X_train(:,2))+mean(X_train(:,3))+mean(X_train(:,4))+mean(X_train(:,5));
%          meanfunc = @meanConst;
%          hyp.mean =1;
    %
    % % hyp for mean linear
    
    %%hyp for poly
%      degree=2;
%      meanfunc = {@meanPoly,degree};
%      hyp.mean = log(ones(degree*(size(X_train,2)-1),1));
    
%     hyp for meanWSPC
%     degree=2;
%     meanfunc = {@meanWSPC,degree};
%       hyp.mean = log(ones(degree*(size(X_train,2)-1)+2*degree,1));
%     
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
                    %        mu2=repmat(mu,size( X_test(:,1:end-1),1),1);
                  %              sigma2=repmat(sigma,size( X_test(:,1:end-1),1),1);
%     
%     
%   
            %              [a, b, c, pred_var, lp, post_local] = gp(hyp2, infFun, meanfunc, ...
           %                       covfunc, likfunc, X_train(:,[1:end-1]), X_train(:,end), (X_test(:,1:end-1)-mu2)./sigma2 ,ones(size(X_test,1), 1));
    
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

