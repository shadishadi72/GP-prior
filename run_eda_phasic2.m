

function [feat_mat,phasic_text]=run_eda_phasic2(x,fs_d)


% feat_mat=mean(x);

data=x;



%%  phasic features 
mean_ph=mean(phasic_text); 
std_ph=std(phasic_text);


%% edasymp
% pband_mean=cal_edasymp(data,fs_d);

      feat_mat=zscore(mean_ph,std_ph);



end
