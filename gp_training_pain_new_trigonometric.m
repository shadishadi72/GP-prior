
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
    
    %     hyp for meanWSPC
    degree=2;
    meanfunc = {@simple_feature_phasic_trig,degree};
    hyp.mean = log(ones((no_feat+2)*degree,1));
    
    
    
    
    
    
    
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


toc


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

