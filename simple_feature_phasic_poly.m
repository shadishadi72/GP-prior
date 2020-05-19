function [m,dm,feat_mat] = simple_feature_phasic_poly(d,hyp,x)
% data=x;
no_feat=8;
%fs_d=2;
% no_feat=size(feat_mat,2);
d = max(abs(floor(d)),1);                              % positive integer degree



if nargin<3, m = int2str(no_feat*d); return; end             % report number of hyperparameters


%%for feature extraction
feat_mat=zeros(size(x,1),no_feat);
fs_d = 32;
for i=1:size(x,1)
    feat=run_eda_phasic(x(i,:),fs_d);
    feat_mat(i,:)=feat;
end



[n,D] = size(feat_mat);
a = reshape(hyp,D,d);

m = zeros(n,1);                                                % allocate memory
for j=1:d, m = m + (feat_mat.^j)*a(:,j); end                            % evaluate mean
dm = @(q) dirder(q,feat_mat,a);                                % directional derivative

end


function dhyp = dirder(q,x,a)
  [D,d] = size(a);
  dhyp = zeros(D*d,1); for j=1:d, dhyp((j-1)*D+(1:D)) = (x.^j)'*q(:); end
end