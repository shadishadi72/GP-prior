function [m,dm,feat_mat] = simple_feature_phasic_partial_feats(feat_group,hyp,x)
% data=x;
no_feat=length(feat_group);
%fs_d=2;
% no_feat=size(feat_mat,2);

fs_d = 32;

if nargin<3, m = int2str(no_feat); return; end             % report number of hyperparameters
% f = mean(x,2);
% function for extracting the features 
% feat_mat=mean(x,2);
%% to disturb with random data 
% feat_mat=randn(size(x,1),no_feat);


%%for feature extraction
feat_mat=zeros(size(x,1),no_feat);
for i=1:size(x,1)
    feat=run_eda_phasic(x(i,:),fs_d);
    feat_mat(i,:)=feat(feat_group);
end

%hyp = varargin{1};
%x = varargin{2};
%hyp(idxs) = 0;
m = feat_mat * hyp(:);

dm =  @(q) feat_mat'*q(:);

end