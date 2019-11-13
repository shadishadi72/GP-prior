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


function feat_mat=run_eda_standard_feat(x)
%  x=X_train_sig(1,:);
data=x;
fs_d=50;


% feat_eda_em_rest_cvx=cell(size(eda_all,1),1);


% for k=1:size(eda_all,1)
  
% data=eda_all{k,1};
%%preprocess 
% data_norm=zscore(data);
% if k==1|| k==7||k==8||k==13
%     fs=200;
% else
%     fs=500;
% end
% fs=500;
% rate=fs/fs_d;
% 
% data_down=downsample(data_norm,rate);

% 
% start=60*fs_d;
% fin=[length(data_down)/(fs_d)-40]*fs_d;

%% choose for each phase 
% 
% start=points(k,7)*fs_d; 
% fin=(points(k,8))*fs_d; 
[r, p, t, l, d, e, obj] = cvxEDA_albe(data,1/fs_d);
data_text=data;
tonic_text=t;
phasic_text=r;
SMNA_text=p;
% SMNA_text_orig=SMNA_text;

%putting to zero 
[a,b]=find(SMNA_text<1);
SMNA_text(a)=0;



%windowing 

%%how many segments , length of t 
win=fs_d*5;
win_ton=fs_d*20;
lap=0.2;
overlap=fs_d*4;
overlap_ton=fs_d*16;

nadvance = win - overlap; 
nrecord  = fix ( (length(data_text)- overlap) / nadvance ); 


nadvance_ton = win_ton - overlap_ton; 
nrecord_ton  = fix ( (length(tonic_text)- overlap_ton) / nadvance_ton ); 

data_text_seg=cell(nrecord,1);
phasic_text_seg=cell(nrecord,1);
SMNA_text_seg=cell(nrecord,1);
tonic_text_seg=cell(nrecord_ton,1);


for i=1:nrecord
  
    if i==1
data_text_seg{i,1}=data_text(1:win);
phasic_text_seg{i,1}=phasic_text(1:win);
SMNA_text_seg{i,1}=SMNA_text(1:win);
    else 
   data_text_seg{i,1}=data_text((i-1)*lap*win : (i-1)*lap*win+win-1);
   phasic_text_seg{i,1}=phasic_text((i-1)*lap*win : (i-1)*lap*win+win-1);
   SMNA_text_seg{i,1}=SMNA_text((i-1)*lap*win : (i-1)*lap*win+win-1);


    end
end

for i=1:nrecord_ton
    if i==1
        tonic_text_seg{i,1}=tonic_text(1:win_ton);
    else 
   tonic_text_seg{i,1}=tonic_text((i-1)*lap*win_ton : (i-1)*lap*win_ton+win_ton-1);
    end 
end


%feature_order=['SMNA(max)';'SMNA(sum)';'SMNA(nom);'phasic(mean)';'phasic(std)';'tonic(mean)';'tonic(std)';'tonic(max)'];


%feature extraction

feat=cell(nrecord,1);
for i=1:nrecord
%% SMNA features
% time=[1:length(SMNA_text_seg{i,1})];
pks= findpeaks(SMNA_text_seg{i,1});

max_pks=max(pks);
sum_pks=sum(pks);
no_pks=length(pks);
%% phasic featues 

mean_ph=mean(phasic_text_seg{i,1});
std_ph=std(phasic_text_seg{i,1});
if isempty(max_pks)==1
    max_pks=NaN;
end
if sum_pks==0
    sum_pks=NaN;
end
feat{i,1}=[max_pks,sum_pks,no_pks,mean_ph,std_ph];
% plot(time,SMNA_text_seg{i,1},locs,pks,'o');
end
feat_mat=cell2mat(feat);
feat_av=nanmean_sh(feat_mat);


f_ton=cell(nrecord_ton,1);
for i=1:nrecord_ton
    ton=tonic_text_seg{i,1};
    %1:max 2:mean 3:std 
    f1=mean(ton);
    f2=std(ton);
    f3=max(ton);
mm=[f1,f2,f3];
f_ton{i,1}=mm;
end
f_ton_av=mean(cell2mat(f_ton));
 

feat_mat=[feat_av,f_ton_av];



end
