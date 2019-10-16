function [X_train,y_train,X_test,y_test] = data_loading_cpt(all_feat_gp)
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
for i=1:no_feat
 
a=X{i,1};
 
rest=a(:,[1:180]);
rest_mean(:,i)=nanmean_sh(rest,2);
end 

% seperating cpt 
cpt_mean=zeros(no_sub,no_feat);
for i=1:no_feat
   
 a=X{i,1};
cpt=a(:,[181:end]);
cpt_mean(:,i)=nanmean_sh(cpt,2);
end 
% a matrix of all data 
all_data=zeros(no_sub*2,no_feat+1);
all_data(1:no_sub,[1:no_feat])=rest_mean;
all_data(no_sub+1:end,[1:no_feat])=cpt_mean;
all_data(1:no_sub,end)=1;
all_data(no_sub+1:end,end)=-1;


% dividing train and test
X_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],[1:no_feat]);
y_train=all_data( [1:no_sub/2,no_sub+1:no_sub+no_sub/2],no_feat+1);

X_test=all_data( [(no_sub/2)+1 : end-no_sub+no_sub/2],[1:no_feat]);
y_test=all_data( [(no_sub/2)+1 : end-no_sub+no_sub/2],no_feat+1);
   


end