function check_manual_and_tool_prediction_agree(test_data,training_data,params_for_gp_toolbox,trainedSystem,S,latent_mu_toolbox,latent_var_toolbox)

gp_trained_params.kernel = 'sqe';
gp_trained_params.mle = false;
gp_trained_params.kernel_params.sigma = params_for_gp_toolbox.sigma;
gp_trained_params.kernel_params.theta_vec = params_for_gp_toolbox.theta_vec;

[Kstar,Kstarstar] = compute_kernel_on_test(test_data,training_data,gp_trained_params);

latent_mu_manual = params_for_gp_toolbox.meanfunc(params_for_gp_toolbox.hyp.mean,test_data) +  Kstar*trainedSystem;

disp('Distance on latent mean and variance between toolbox and matrix computation')
disp(max(abs(latent_mu_manual - latent_mu_toolbox)))

latent_var_manual = Kstarstar - Kstar * S * Kstar';
disp(max(abs(diag(latent_var_manual) - latent_var_toolbox)))

end