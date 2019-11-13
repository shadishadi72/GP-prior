function [m,dm] = linear_excluding(idxs,hyp,x)
if nargin<3, m = 'D'; return; end             % report number of hyperparameters
%hyp = varargin{1};
%x = varargin{2};
hyp(idxs) = 0;
m = x * hyp(:);

dm =  @(q) x'*q(:);

end