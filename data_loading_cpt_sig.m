
function [X_train_sig,y_train,X_test_sig,y_test] = data_loading_cpt_sig(eda_cp_red_mat,eda_cp_rest_red_mat);

% eda signal as input 
divide_train=12;
no_sub=size(eda_cp_red_mat,1);
divide_test=no_sub-divide_train;

rate=50/1;
len=size(eda_cp_red_mat,2);
cp_tr=eda_cp_red_mat(1:divide_train,:);
rest_tr=eda_cp_rest_red_mat(1:divide_train,:);
X_train_sig=[cp_tr;rest_tr];
X_train_sig=X_train_sig';
% X_train_sig=downsample(X_train_sig,rate);
X_train_sig=X_train_sig';

y_train(1:divide_train)=1;
y_train(divide_train+1:divide_train*2)=-1;
y_train=y_train';  

cp_tes=eda_cp_red_mat(divide_train+1:end,:);
rest_tes=eda_cp_rest_red_mat(divide_train+1:end,:);
X_test_sig=[cp_tes;rest_tes];
X_test_sig=X_test_sig';
% X_test_sig=downsample(X_test_sig,rate);
X_test_sig=X_test_sig';

y_test(1:divide_test)=1;
y_test(divide_test+1:divide_test*2)=-1;
y_test=y_test';

end