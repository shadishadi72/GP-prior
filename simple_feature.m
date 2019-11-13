function [m,dm,feat_mat] = simple_feature(hyp,x)
% data=x;

% no_feat=size(feat_mat,2);
no_feat=8;

if nargin<2, m = int2str(no_feat); return; end             % report number of hyperparameters
% f = mean(x,2);
% function for extracting the features 
% feat_mat=mean(x,2);
feat_mat=zeros(size(x,1),no_feat);
for i=1:size(x,1)
feat=run_eda_standard_feat(x(i,:));
feat_mat(i,:)=feat;
end

%hyp = varargin{1};
%x = varargin{2};
%hyp(idxs) = 0;
m = feat_mat * hyp(:);

dm =  @(q) feat_mat'*q(:);

end