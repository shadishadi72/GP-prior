function [X_train,y_train,X_test,Y_test] = data_loading_cpt(all_feat_gp)
X{1,:}=all_feat_gp.mu;
X{2,:}=all_feat_gp.var;
X{3,:}=all_feat_gp.lf;
X{4,:}=all_feat_gp.hf;
X{5,:}=all_feat_gp.lfhf;
X{6,:}=all_feat_gp.edasymp;
X{7,:}=all_feat_gp.edahf;

no_feat=7;
no_sub=26;

% sepearating rest
rest_mean=zeros(no_sub,no_feat);

% for time series
% rest_all=cell(no_sub,no_feat);

for i=1:no_feat
    for j=1:no_sub
        
        a=X{i,1};
        
        rest=a(:,[120:180]);
%         rest=a(:,[60:180]); not good results
        
        
        rest_mean(:,i)=nanmean_sh(rest,2);
        %     % for time series
        %     rest_all{j,i}=rest(no_sub,:);
        
    end
    
end

% seperating cpt
cpt_mean=zeros(no_sub,no_feat);

%     % for time series
% cpt_all=cell(no_sub,no_feat);

for i=1:no_feat
    for j=1:no_sub
        
        a=X{i,1};
        cpt=a(:,[181:end-120]);
        %  cpt=a(:,[181:end-60]); not good results
        
        cpt_mean(:,i)=nanmean_sh(cpt,2);
        %     % for time series
        %       cpt_all{j,i}=cpt(no_sub,:);
        
    end
    
    
end
% a matrix of all data
all_data=zeros(no_sub*2,no_feat+1);
all_data(1:no_sub,[1:no_feat])=rest_mean;
all_data(no_sub+1:end,[1:no_feat])=cpt_mean;
all_data(1:no_sub,end)=1;
all_data(no_sub+1:end,end)=-1;



%% input for averaged signal
% dividing train and test
X_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],[1:no_feat]);
y_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],no_feat+1);

X_test=all_data( [(no_sub/2)+1 : no_sub , end-no_sub+no_sub/2+1 :end],[1:no_feat]);
Y_test=all_data( [(no_sub/2)+1 : no_sub , end-no_sub+no_sub/2+1 :end],no_feat+1);


%
% %% input for time series
% % for time series data
% % train
% X_train=cell(no_sub,no_feat);
% y_train=zeros(no_sub,1);
% % class 1 rest
% X_train(1:no_sub/2,1:no_feat)=rest_all(1:no_sub/2,:);
% y_train(1:no_sub/2,1)=1;
% % calss 2 cpt
% X_train(no_sub/2+1:no_sub,1:no_feat)=cpt_all(1:no_sub/2,:);
% y_train(no_sub/2+1:no_sub,1)=-1;
%
%
% % test
% X_test=cell(no_sub,no_feat);
% y_test=zeros(no_sub,1);
% % class 1 rest
% X_test(1:no_sub/2,1:no_feat)=rest_all(no_sub/2+1:no_sub,:);
% y_test(1:no_sub/2,1)=1;
% % calss 2 cpt
% X_test(no_sub/2+1:no_sub,1:no_feat)=cpt_all(no_sub/2+1:no_sub,:);
% y_test(no_sub/2+1:no_sub,1)=-1;







end