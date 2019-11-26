function [X_train,y_train,X_test,y_test] = data_loading_pain_mc(inp);

data=inp;
% 5 heat level for each subject 
% choose only min and max heat stimuli ( 0 and 5 ) 
% 0,2,4 72% accuracy
heat1=find(data(:,9)==0);
data_heat1=data(heat1,:);

heat2=find(data(:,9)==1);
data_heat2=data(heat2,:);

heat3=find(data(:,9)==2);
data_heat3=data(heat3,:);

heat4=find(data(:,9)==3);
data_heat4=data(heat4,:);

heat5=find(data(:,9)==4);
data_heat5=data(heat5,:);

class1=data_heat1(:,[2:8]);
class2=data_heat2(:,[2:8]);
class3=data_heat3(:,[2:8]);
class4=data_heat4(:,[2:8]);
class5=data_heat5(:,[2:8]);

no_sub=size(class1,1);
no_feat=size(class1,2);
%% division 
% train : half class1 half class2 half class3 
% test: half class1 half class2 half class3 


% divide_train=floor(no_sub/2);
divide_train=floor(no_sub/2);
divide_test=no_sub-divide_train;
no_class=5;
%%% train %% 
X_train=zeros(divide_train*no_class,no_feat);
y_train=zeros(divide_train*no_class,no_class);
% class 1 
X_train(1:divide_train,1:no_feat)=class1(1:divide_train,:);
y_train(1:divide_train,1)=1;
y_train(1:divide_train,2)=0;
y_train(1:divide_train,3)=0;
y_train(1:divide_train,4)=0;
y_train(1:divide_train,5)=0;




% class 2 
X_train(divide_train+1:divide_train*2,1:no_feat)=class2(1:divide_train,:);
y_train(divide_train+1:divide_train*2,1)=0;
y_train(divide_train+1:divide_train*2,2)=1;
y_train(divide_train+1:divide_train*2,3)=0;
y_train(divide_train+1:divide_train*2,4)=0;
y_train(divide_train+1:divide_train*2,5)=0;



% class 3 
X_train(divide_train*2+1:divide_train*3,1:no_feat)=class3(1:divide_train,:);
y_train(divide_train*2+1:divide_train*3,1)=0;
y_train(divide_train*2+1:divide_train*3,2)=0;
y_train(divide_train*2+1:divide_train*3,3)=1;
y_train(divide_train*2+1:divide_train*3,4)=0;
y_train(divide_train*2+1:divide_train*3,5)=0;



% class 4 
X_train(divide_train*3+1:divide_train*4,1:no_feat)=class4(1:divide_train,:);
y_train(divide_train*3+1:divide_train*4,1)=0;
y_train(divide_train*3+1:divide_train*4,2)=0;
y_train(divide_train*3+1:divide_train*4,3)=0;
y_train(divide_train*3+1:divide_train*4,4)=1;
y_train(divide_train*3+1:divide_train*4,5)=0;


% class 5 
X_train(divide_train*4+1:divide_train*5,1:no_feat)=class5(1:divide_train,:);
y_train(divide_train*4+1:divide_train*5,1)=0;
y_train(divide_train*4+1:divide_train*5,2)=0;
y_train(divide_train*4+1:divide_train*5,3)=0;
y_train(divide_train*4+1:divide_train*5,4)=0;
y_train(divide_train*4+1:divide_train*5,5)=1;


%%% test %% 
X_test=zeros(divide_test*no_class,no_feat);
y_test=zeros(divide_test*no_class,no_class);
% class 1 
X_test(1:divide_test,1:no_feat)=class1(divide_train+1:no_sub,:);
y_test(1:divide_test,1)=1;
y_test(1:divide_test,2)=0;
y_test(1:divide_test,3)=0;
y_test(1:divide_test,4)=0;
y_test(1:divide_test,5)=0;




% class 2 
X_test(divide_test+1:divide_test*2,1:no_feat)=class2(divide_train+1:no_sub,:);
y_test(divide_test+1:divide_test*2,1)=0;
y_test(divide_test+1:divide_test*2,2)=1;
y_test(divide_test+1:divide_test*2,3)=0;
y_test(divide_test+1:divide_test*2,4)=0;
y_test(divide_test+1:divide_test*2,5)=0;



% class 3 
X_test(divide_test*2+1:divide_test*3,1:no_feat)=class3(divide_train+1:no_sub,:);
y_test(divide_test*2+1:divide_test*3,1)=0;
y_test(divide_test*2+1:divide_test*3,2)=0;
y_test(divide_test*2+1:divide_test*3,3)=1;
y_test(divide_test*2+1:divide_test*3,4)=0;
y_test(divide_test*2+1:divide_test*3,5)=0;



% class 4 
X_test(divide_test*3+1:divide_test*4,1:no_feat)=class4(divide_train+1:no_sub,:);
y_test(divide_test*3+1:divide_test*4,1)=0;
y_test(divide_test*3+1:divide_test*4,2)=0;
y_test(divide_test*3+1:divide_test*4,3)=0;
y_test(divide_test*3+1:divide_test*4,4)=1;
y_test(divide_test*3+1:divide_test*4,5)=0;


% class 5 
X_test(divide_test*4+1:divide_test*5,1:no_feat)=class5(divide_train+1:no_sub,:);
y_test(divide_test*4+1:divide_test*5,1)=0;
y_test(divide_test*4+1:divide_test*5,2)=0;
y_test(divide_test*4+1:divide_test*5,3)=0;
y_test(divide_test*4+1:divide_test*5,4)=0;
y_test(divide_test*4+1:divide_test*5,5)=1;
end

