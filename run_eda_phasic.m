% % get all eda files
% addpath('/Users/shadi/Documents/blood text dataset')
% datalist=dir('/Users/shadi/Documents/blood text dataset/Aquila di sangue/Biopac');
% datalist = {datalist.name};
% datalist=datalist(1,[3:end]);
% 
% eda_bt_all=cell(size(datalist,2),1);
% 
% 
% 
% for i=1%:size(datalist,2)
%     eda=load_acq(char(datalist(1,i)));
%     eda2=eda.data(:,3);
%     eda_bt_all{i,1}=eda;
% end
% % 
% % save eda_bt_all eda_bt_all 


function feat_mat=run_eda_phasic(x,fs_d)


% feat_mat=mean(x);

data=x;
 [r, p, t, l, d, e, obj] = cvxEDA_albe(data,1/fs_d);
% data_text=data;
tonic_text=t;
phasic_text=r;
SMNA_text=p;

% check 
% figure
% plot(data)
% hold on 
% plot(tonic_text);
% plot(phasic_text);
% 
% 
%%  phasic features 
mean_ph=mean(phasic_text);
std_ph=std(phasic_text);
%%  SMNA features
%putting to zero 
[a,b]=find(SMNA_text<1);
SMNA_text(a)=0;
pks= findpeaks(SMNA_text);
% 
 max_pks=max(pks);
sum_pks=sum(pks);
no_pks=length(pks);
if isempty(max_pks)==1
    max_pks=0;
end
if isempty(no_pks)==1
    no_pks=0;
end
% %% tonic features  
mean_t=mean(tonic_text);
std_t=std(tonic_text);

%% edasymp
pband_mean=cal_edasymp(data,fs_d);

%    feat_mat=[mean_ph,std_ph,max_pks,sum_pks,no_pks,mean_t,std_t,pband_mean];
     feat_mat=[mean_ph,std_ph,max_pks,sum_pks,no_pks];
%       feat_mat=[mean_t,std_t];


    feat_mat=zscore(feat_mat);


end
