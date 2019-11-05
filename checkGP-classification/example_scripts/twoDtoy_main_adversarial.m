run('toolboxes/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc

addpath(genpath('utils/'))

%Code parameters
savingFlag = true;
numberOfThreads = 1;

%GP model parameters
likfunc = @likErf; %link function
infFun = @infEP; %inference method
num_trs = 1000; %number of training points
num_tes = 200; %number of test points
train_iter = 40;


%Branch and bound parameters
maxiters = 10000; %maximum number of branch and bound iterations
tollerance = 0.02; %bound tollerance required
epsilons = [0,0.2];
Testpoints = [1:50];
n_eps = length(epsilons);
bound_comp_opts.mod_modus = 'ls';
bound_comp_opts.pix_2_mod = [];
bound_comp_opts.constrain_2_one = false;
bound_comp_opts.max_iterations = maxiters;
bound_comp_opts.tollerance = tollerance;
bound_comp_opts.N = 10000;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-20;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.likmode = 'analytical';
bound_comp_opts.mode = 'binarypi';
%Adversarial example Parameters
step_size = 0.001;
n_steps = 1;

%%%

rng(1)
maxNumCompThreads(numberOfThreads);
dirName = strcat('results/',datestr(datetime('now')),'_2DToyAdversarial');
if exist(dirName,'dir') ~=7
    if savingFlag
        mkdir(dirName)
    end
end



global mu_time
global std_time
global discrete_time
global inference_time
global pred_var

global post
discrete_time = 0;
mu_time = 0;
std_time = 0;
inference_time = 0;



[X_train,y_train,X_test,y_test] = generate_2d_synthetic_datasets(num_trs,num_tes);



%% training of the GP
disp('Training GP')
meanfunc = @meanZero;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));

[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,num_trs,infFun,meanfunc,covfunc,likfunc);

disp('Done with training')
%%

if bound_comp_opts.numberOfThreads > 1
    if isempty(gcp('nocreate'))
        parpool(bound_comp_opts.numberOfThreads);
    end
end



%%Praparing results structure and matrixes
defense.analytical = zeros(1+length(Testpoints),n_eps);
defense.analytical(2:end,1) = exp(lp(Testpoints));
defense.analytical(1,:) = epsilons;
defense.analytical_exact = zeros(1+length(Testpoints),n_eps);
defense.analytical_exact(2:end,1) = exp(lp(Testpoints));
defense.analytical_exact(1,:) = epsilons;
iter_count.analytical = zeros(1+length(Testpoints),n_eps);
iter_count.analytical(1,:) = epsilons;
flags.analytical = zeros(1+length(Testpoints),n_eps);
flags.analytical(1,:) = epsilons;

attacks_GPFGS.probs = zeros(1+length(Testpoints),n_eps);
attacks_GPFGS.probs(2:end,1) = exp(lp(Testpoints));
attacks_GPFGS.probs(1,:) = epsilons;
attacks_GPJM.probs = zeros(1+length(Testpoints),n_eps);
attacks_GPJM.probs(2:end,1) = exp(lp(Testpoints));
attacks_GPJM.probs(1,:) = epsilons;

min_breakable_vs_broken = [ones(length(Testpoints),1),ones(length(Testpoints),1),ones(length(Testpoints),1),ones(length(Testpoints),1)];


%% For loop over test points
global training_data
global training_labels
global loop_vec2
training_data = X_train;
training_labels = y_train;
loop_vec2 = discretise_real_line(bound_comp_opts.N);
clear X_train
clear Kstar Kstarstar data latent_variance_prediction

S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

ccount = 1;

for dd = 1:length(Testpoints)
    
    no_bound_crossing = true;
    no_bound_crossing_d = true;
    
    testIdx = Testpoints(dd);
    testPoint = X_test(testIdx,:);
    
    %  For loop over epsilons to get pi_Ls and pi_Us
    
    pi_LLs_a = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
    pi_UUs_a = [exp(lp(testIdx)),2:n_eps];
    pi_LUs_a = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
    pi_ULs_a = [exp(lp(testIdx)),2:n_eps];
    
    pi_LLs_d = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
    pi_UUs_d = [exp(lp(testIdx)),2:n_eps];
    pi_LUs_d = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
    pi_ULs_d = [exp(lp(testIdx)),2:n_eps];
    
    pi_Ls_s = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
    pi_Us_s = [exp(lp(testIdx)),2:n_eps];
        
    [~,bound_comp_opts.pix_2_mod] = maxk(params_for_gp_toolbox.theta_vec,2);
    
    
    for ii = 2:n_eps
        bound_comp_opts.epsilon = epsilons(ii);
        disp('Current epsilon')
        disp(bound_comp_opts.epsilon)
        
        %%-------- Attacks --------
        
        %%1. GPFGS
        [attacks_GPFGS.probs(dd+1,ii),class1,maxminmode] = GPFGS_attack(lp(testIdx),epsilons(ii),step_size,testPoint,n_steps,params_for_gp_toolbox,bound_comp_opts);
        
        if class1 && (attacks_GPFGS.probs(dd+1,ii) < 0.5) && (min_breakable_vs_broken(dd,2) == 1)
            min_breakable_vs_broken(dd,2) = epsilons(ii);
        elseif ~class1 && ( attacks_GPFGS.probs(dd+1,ii) >= 0.5) && (min_breakable_vs_broken(dd,2) == 1)
            min_breakable_vs_broken(dd,2) = epsilons(ii);
        end
        
        
        
        %%2. GPJM
        [out,class1] = GPJM_attack(testPoint,n_steps,epsilons(ii),lp(testIdx),params_for_gp_toolbox,bound_comp_opts,step_size);
        attacks_GPJM.probs(dd+1,ii) = out;
        if class1 && (attacks_GPJM.probs(dd+1,ii) < 0.5) && (min_breakable_vs_broken(dd,3) == 1)
            min_breakable_vs_broken(dd,3) = epsilons(ii);
        elseif ~class1 && (attacks_GPJM.probs(dd+1,ii) >= 0.5) && (min_breakable_vs_broken(dd,3) == 1)
            min_breakable_vs_broken(dd,3) = epsilons(ii);
        end

        
        %%-------- Defenses --------
        
        
        [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
            bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
        bound_comp_opts.x_L = x_L;
        bound_comp_opts.x_U = x_U;
        
        
        if analyticalFlag && no_bound_crossing
            flags.analytical(dd+1,1) = testIdx;
            iter_count.analytical(dd+1,1) = testIdx;
            aux = tic;
            [pi_LL,pi_UU, pi_LU,pi_UL,count,exitFlag] = main_pi_hat_computation(maxminmode,testPoint,testIdx,...
                params_for_gp_toolbox,bound_comp_opts,trainedSystem,S);
            pi_LLs_a(ii) = pi_LL;
            pi_UUs_a(ii) = pi_UU;
            pi_LUs_a(ii) = pi_LU;
            pi_ULs_a(ii) = pi_UL;
            toc(aux)
            if class1
                defense.analytical(dd+1,ii) = pi_LL;
                defense.analytical_exact(dd+1,ii) = pi_LU;
                iter_count.analytical(dd+1,ii) = count.min;
                flags.analytical(dd+1,ii) = exitFlag.min;
            else
                defense.analytical(dd+1,ii) = pi_UU;
                defense.analytical_exact(dd+1,ii) = pi_UL;
                iter_count.analytical(dd+1,ii) = count.max;
                flags.analytical(dd+1,ii) = exitFlag.max;
            end
        end
      
        
        
        if class1 && (pi_LLs_a(ii) < 0.5) && (min_breakable_vs_broken(dd,1) == 1)
            min_breakable_vs_broken(dd,1) = epsilons(ii);
            no_bound_crossing = false;
        elseif ~class1 && (pi_UUs_a(ii) >= 0.5) && (min_breakable_vs_broken(dd,1) == 1)
            min_breakable_vs_broken(dd,1) = epsilons(ii);
            no_bound_crossing = false;
        end
        
        
    end
    
    disp(strcat('DONE WITH TESTPOINT ', num2str(testIdx)))
    
end

min_breakable_vs_broken(min_breakable_vs_broken == 1) = max(epsilons) + 0.01;


%% Save parameters and settings used

saveables.modelAcc = acc;
saveables.boundOpts = bound_comp_opts;
saveables.covfunc = covfunc;
saveables.infFun = infFun;
saveables.likfunc = likfunc;
saveables.meanfunc = meanfunc;
saveables.n_samples = n_samples;
saveables.num_tes = num_tes;
saveables.num_trs = num_trs;
saveables.params_for_gp_toolbox = params_for_gp_toolbox;
saveables.Testpoints = Testpoints;
saveables.GPFGS = attacks_GPFGS;
saveables.GPJM = attacks_GPJM;
saveables.breaking_analysis = min_breakable_vs_broken;
saveables.defenses = defense;
saveables.iter_count = iter_count;
saveables.exitFlags = flags;


if savingFlag
    save(strcat(dirName,'/ParamsAndSettings.mat'),'saveables');
    
    disp('Parameters and settings succesfully saved')
end


%%

tempi = ((min_breakable_vs_broken(:,1) > min_breakable_vs_broken(:,2)) + (min_breakable_vs_broken(:,1) > min_breakable_vs_broken(:,3)));
if sum(tempi) == 0
    disp('No robustness guarantee was broken')
else
    disp('At least one robustness guarantee was broken. Closer inspection needed!')
end


%% Plot 2 points
for jj = 2:3;
    for ii = 1:length(epsilons)
        if discreteFlag
            if defense.discretised(jj,ii) == 0
                defense.discretised(jj,ii) = defense.discretised(jj,ii-1)-sign(defense.discretised(jj,1)-0.5)*0.05;
                defense.discretised_exact(jj,ii) = defense.discretised_exact(jj,ii-1)-sign(defense.discretised_exact(jj,1)-0.5)*0.05;
            end
        end
        if analyticalFlag
            if defense.analytical(jj,ii) == 0
                defense.analytical(jj,ii) = defense.analytical(jj,ii-1)-sign(defense.analytical(jj,1)-0.5)*0.05;
                defense.analytical_exact(jj,ii) = defense.analytical_exact(jj,ii-1)-sign(defense.analytical_exact(jj,1)-0.5)*0.05;
            end
        end
    end
end


%%
%%
boundary = 0.5*ones(1,length(epsilons));

fig = figure;
ax1 = subplot(1,2,1);
plot(epsilons,attacks_GPFGS.probs(2,:),'r','DisplayName','GPFGS attack')
hold on
plot(epsilons,attacks_GPJM.probs(2,:),'r--','DisplayName','GPJM attack')
hold on
plot(epsilons,boundary,'m','HandleVisibility','off')
hold on
%legend(legendplot,strcat('sampled w',n_samples))
if defense.analytical(2,1) < 0.5
    plot(epsilons,defense.analytical(2,:),'b','DisplayName','upper bound maximum')
    hold on
    plot(epsilons,defense.analytical_exact(2,:),'b--','DisplayName','lower bound maximum')
    hold off
    ylim([0 0.55])
    legend('Location','southeast')
else
    ylim([0.45 1.0])
    legend('Location','northeast')
    plot(epsilons,defense.analytical(2,:),'b','DisplayName','lower bound minimum')
    hold on
    plot(epsilons,defense.analytical_exact(2,:),'b--','DisplayName','upper bound minimum')
    hold off
end
xlim([0 max(epsilons)])
title(strcat('Testpoint',num2str(Testpoints(1))))
xlabel('epsilon')
ylabel('pi')

ax2 = subplot(1,2,2);
plot(epsilons,attacks_GPFGS.probs(3,:),'r','DisplayName','GPFGS attack')
hold on
plot(epsilons,attacks_GPJM.probs(3,:),'r--','DisplayName','GPJM attack')
hold on
plot(epsilons,boundary,'m','HandleVisibility','off')
hold on
if defense.analytical(3,1) < 0.5
    plot(epsilons,defense.analytical(3,:),'b','DisplayName','upper bound maximum')
    hold on
    plot(epsilons,defense.analytical_exact(3,:),'b--','DisplayName','lower bound maximum')
    hold off
    ylim([0 0.55])
    xlim([0 max(epsilons)])
    legend('Location','southeast')
else
    plot(epsilons,defense.analytical(3,:),'b','DisplayName','lower bound of minimum')
    hold on
    plot(epsilons,defense.analytical(3,:),'b--','DisplayName','upper bound of minimum')
    hold off
    ylim([0.45 1.0])
    xlim([0 max(epsilons)])
    legend('Location','northeast')
end
title(strcat('Testpoint',num2str(Testpoints(2))))
xlabel('epsilon')
ylabel('pi')


if savingFlag
    saveas(fig,strcat(dirName,'/AdversarialAnalysis_Testpoints',num2str(Testpoints(1)),'_and_',...
        num2str(Testpoints(2)),'.png'))
end