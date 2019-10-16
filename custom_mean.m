function [m,dm] = custom_mean(hyp, x)
if nargin<2, m = 'D'; return; end             % report number of hyperparameters


m = x*hyp(:);

dm =  @(q) x'*q(:);

end