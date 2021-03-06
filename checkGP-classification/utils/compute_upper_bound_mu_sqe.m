function [mu_lb,x_mu_lb] = compute_upper_bound_mu_sqe(trainedSystem,x_L,x_U,theta_vec,...
                                sigma,z_i_L_vec,z_i_U_vec,meanfunc,hyp_mean,minValue,feats_extrema)
[mu_lb,x_mu_lb] = compute_lower_bound_mu_sqe(-trainedSystem,x_L,x_U,theta_vec,...
    sigma,z_i_L_vec,z_i_U_vec,meanfunc,-hyp_mean,feats_extrema);

mu_lb= - mu_lb;
%if nargin >=7
mu_lb = max(mu_lb,minValue);
end