function [X_train,y_train,X_test,y_test] = data_loading_pain(inp);

data=inp;
% 5 heat level for each subject 
% choose only min and max heat stimuli ( 0 and 5 ) 
min_heat=find(data(:,9)==0);
data_min_heat=data(min_heat,:);

max_heat=find(data(:,9)==4);
data_max_heat=data(max_heat,:);

class1=data_min_heat(:,[2:8]);
class2=data_max_heat(:,[2:8]);



no_sub=size(class1,1);
no_feat=size(class1,2);
%% division 

divide_train=floor(no_sub/2);
divide_test=no_sub-divide_train;

%%% train %% 
X_train=zeros(divide_train*2,no_feat);
y_train=zeros(divide_train*2,1);
% class 1
X_train(1:divide_train,1:no_feat)=class1(1:divide_train,:);
y_train(1:divide_train,1)=1;
% calss 2 
X_train(divide_train+1:divide_train*2,1:no_feat)=class2(1:divide_train,:);
y_train(divide_train+1:divide_train*2,1)=-1;

%%test%%
X_test=zeros(divide_test*2,no_feat);
y_test=zeros(divide_test*2,1);
% class 1 
X_test(1:divide_test,1:no_feat)=class1(divide_train+1:no_sub,:);
y_test(1:divide_test,1)=1;
% calss 2 
X_test(divide_test+1:divide_test*2,1:no_feat)=class2(divide_train+1:no_sub,:);
y_test(divide_test+1:divide_test*2,1)=-1;
