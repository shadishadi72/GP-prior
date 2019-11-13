function [X_train,y_train,X_test,y_test] = data_loading_cpt_cvx(cp_cvx,rest_cvx,cold_new_zscore,cold_rest_new_zscore)


% features to use 
%SMNA(max),SMNA(nom),'tonic(mean)','edasymp'
%feature_order=['SMNA(max)';'SMNA(sum)';'SMNA(nom)';'phasic(mean)';'phasic(std)';'tonic(mean)';'tonic(std)';'tonic(max)'];
cp=cell2mat(cp_cvx);
rest=cell2mat(rest_cvx);

% reduce 



%adding edasymp 
cp_edasymp=cell2mat(cold_new_zscore')';
% cp_edasymp=cp_edasymp(:,[20:70]); % 
 cp_edasymp_mean=nanmean_sh(cp_edasymp,2);
% for i=1:size(cp_edasymp,1)
% cp_edasymp_mean(i,1)=max(cp_edasymp(i,:));
% end


rest_edasymp=cell2mat(cold_rest_new_zscore')';
% rest_edasymp=rest_edasymp(:,[120:170]); % 
rest_edasymp_mean=nanmean_sh(rest_edasymp,2);
% for i=1:size(rest_edasymp,1)
% rest_edasymp_mean(i,1)=max(rest_edasymp(i,:));
% end


% adding to fetaure list 
cp(:,9)=cp_edasymp_mean;
rest(:,9)=rest_edasymp_mean;

% to make eda signals and features the same subjects 
% cp=cp([2:5,7:10,12:17,19:21,23:end],:);
% rest=rest([2:5,7:10,12:17,19:21,23:end],:);

% excluding subjects by plot checking 
% cp=cp([1:5,7,9,11:18,20,22:24,28],:);

% rest=rest([1:5,7,9,11:18,20,22:24,28],:);





no_feat=size(rest,2);
no_sub=size(rest,1);

% 
% 
%     %% input for averaged signal
%     % dividing train and test
%     X_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],[1:no_feat]);
%     y_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],no_feat+1);
% 
%     X_test=all_data( [(no_sub/2)+1 : end-no_sub+no_sub/2],[1:no_feat]);
%     y_test=all_data( [(no_sub/2)+1 : end-no_sub+no_sub/2],no_feat+1);
%
%% division 

divide_train=no_sub/2;
divide_test=no_sub-divide_train;

X_train=zeros(divide_train*2,no_feat);
y_train=zeros(divide_train*2,1);
% class 1 rest
X_train(1:divide_train,1:no_feat)=cp(1:divide_train,:);
y_train(1:divide_train,1)=1;
% calss 2 cpt
X_train(divide_train+1:divide_train*2,1:no_feat)=rest(1:divide_train,:);
y_train(divide_train+1:divide_train*2,1)=-1;


X_test=zeros(divide_test*2,no_feat);
y_test=zeros(divide_test*2,1);
% class 1 rest
X_test(1:divide_test,1:no_feat)=cp(divide_train+1:no_sub,:);
y_test(1:divide_test,1)=1;
% calss 2 cpt
X_test(divide_test+1:divide_test*2,1:no_feat)=rest(divide_train+1:no_sub,:);
y_test(divide_test+1:divide_test*2,1)=-1;







end