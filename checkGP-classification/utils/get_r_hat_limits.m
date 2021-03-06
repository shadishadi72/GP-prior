function [r_L_hat,r_U_hat] = get_r_hat_limits(r_L,r_U,B,a_vec,x_L,x_U)
% r_hat = U'*r. Given rectangular bounds for r, I here compute
% corresponding enclosing rectangular bounds for r_hat.

global U
%r_star_L = zeros(length(r_L),1);
%r_star_U = zeros(length(r_U),1);

%r_L_hat_old = zeros(size(r_L));
%r_U_hat_old = zeros(size(r_L));

r_L_hat = zeros(size(r_L));
r_U_hat = zeros(size(r_L));



m = length(x_L);
n = length(r_L);

%opts.Algorithm = 'interior-point';
%[x_opt,lb,exitFlag] = linprog(b_lin_obj_vec,B,-a_vec,[],[],[x_L,r_L_hat],[x_U,r_U_hat],opts);

opts = optimoptions('linprog');
opts.Display = 'off';
opts.Algorithm = 'interior-point-legacy';
opts.Preprocess = 'none';
opts.OptimalityTolerance = 1e-2;
opts.ConstraintTolerance = 1e-16;

nn = 1;
a_vec_hat = - a_vec - B(:,1:m)*x_L' - B(:,(m+1):end)*r_L';
B(:,1:m) = B(:,1:m).*((x_U - x_L)/nn);
B(:,(m+1):end) = B(:,(m+1):end).*((r_U - r_L)/nn);

%aux = 0;

ps = parallel.Settings;
ps.Pool.AutoCreate = false;

parfor ii = 1:n
    %ii
    u_i = U(:,ii);
    %neg_idxs = u_i < 0;
    %pos_idxs = u_i >= 0;
    %r_star_L(pos_idxs) =  r_L(pos_idxs);
    %r_star_L(neg_idxs) =  r_U(neg_idxs);
    %r_star_U(pos_idxs) =  r_U(pos_idxs);
    %r_star_U(neg_idxs) =  r_L(neg_idxs);
    %r_L_hat_old(ii) = u_i'*r_star_L;
    %r_U_hat_old(ii) = u_i'*r_star_U;
    
    %f = [zeros(m,1);u_i];
    
    
    %[~,idxs] = maxk(abs(u_i),500);
    %try
    %idxs = [2*idxs;2*idxs-1];
 
    f_hat = [zeros(m,1);u_i.*((r_U - r_L)/nn)'];
    [~,r_L_hat_hat,~] = linprog(f_hat,B,a_vec_hat,[],[],zeros(m+n,1),nn*ones(m+n,1),opts);
    r_L_hat(ii) = r_L_hat_hat + r_L*u_i;
    
    
    %status
    [~,r_U_hat_temp,~] = linprog(-f_hat,B,a_vec_hat,[],[],zeros(m+n,1),nn*ones(m+n,1),opts);
    %status
    r_U_hat(ii) = - r_U_hat_temp + r_L*u_i;
    

end


%disp(aux)






end