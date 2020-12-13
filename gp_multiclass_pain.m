% clear all %#ok<CLALL>
% close all
clc
rng(1)
addpath(genpath('utils/'))
%addpath('plotScripts')
addpath(genpath('toolbox/gpstuff-master/'))

%GP model parameters
num_trs = 500;
num_tes = 500;


%Code parameters
savingFlag = true;
plottingFlag = true;
numberOfThreads = 1;
gp_name_id = ['cache/','mnist358_',int2str(num_trs),'.mat'];


%Branch and bound parameters
discreteFlag = true;
statisticalFlag = false;
epsilons = [0,0.125];
n_eps = length(epsilons);
Testpoints = [1];
bound_comp_opts.constrain_2_one = true;
bound_comp_opts.max_iterations = 50;
bound_comp_opts.tollerance = 0.02;
bound_comp_opts.N = 100;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-8;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.likmode = 'discretised';
bound_comp_opts.mode = 'multipi';




%some global variables used for time monitoring of the method
global mu_time
global std_time
global discrete_time
global inference_time
discrete_time = 0;
mu_time = 0;
std_time = 0;
inference_time = 0;




maxNumCompThreads(numberOfThreads);
dirName = strcat('results/',datestr(datetime('now')),'_MNIST_multiclass_interpretability');
if exist(dirName,'dir') ~=7
    if savingFlag
        mkdir(dirName)
    end
end

[X_train,y_train,X_test,y_test ] = load_mnist358(num_trs,num_tes);

%% load pain data 
% load('dataset_pain_1234.mat')
% load('dataset_pain_1234_tar.mat')

dataset_fin=dataset_pain_024;
dataset_tar=dataset_pain_024_tar;
% % create 3 class dataset
% dataset_fin=dataset_pain_1234([1:end-87],:);
% dataset_tar=dataset_pain_1234_tar([1:end-87],[1:3]);



fs=32;
fs_d=32;
rate_down=fs/fs_d;

dataset_down=downsample(dataset_fin(:,[2:end])',rate_down)';
dataset_fin2=zeros(size(dataset_fin,1),size(dataset_down,2)+1);
dataset_fin2(:,1)=dataset_fin(:,1);
dataset_fin2(:,[2:end])=dataset_down;
dataset_fin=dataset_fin2;







subj=dataset_fin(:,1);
dataset_fin=dataset_fin(:,[2:end]);
all_pred=[];
all_label=[];

%% LOSO 
for ii=1:max(subj)
    disp( ii)
 
    class_score=0;

    
    %% for LOSO
    %
    ix=find(subj==ii);
    trainingSet=dataset_fin;
    targetset=dataset_tar;
    X_test=trainingSet(ix,:);
    y_test=targetset(ix,:);
    
    trainingSet(ix,:)=[];
     targetset(ix,:)=[];
    X_train=trainingSet(:,:);
    y_train=targetset(:,:);



%% GP Training
if ~isfile(gp_name_id)
    disp('Training the GP - could take a few minutes...')
    % Create covariance functions
    gpcf1 = gpcf_sexp('lengthScale', ones(1,size(X_train,2)), 'magnSigma2', 1);
    % Set the prior for the parameters of covariance functions
    pl = prior_t();
    pm = prior_sqrtt();
    gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pm);
    % Create the GP structure
    gp = gp_set('lik', lik_softmax, 'cf', gpcf1, 'jitterSigma2', 1e-2);
    % ------- Laplace approximation --------
    fprintf(['Softmax model with Laplace integration over the latent\n' ...
        'values and MAP estimate for the parameters\n'])
    % Set the approximate inference method
    gp = gp_set(gp, 'latent_method', 'Laplace');
    % Set the options for the optimization
    opt=optimset('TolFun',0.1,'TolX',1e-3,'Display','iter','MaxIter',10);
    % Optimize with the BFGS quasi-Newton method
    gp=gp_optim(gp,X_train,y_train,'opt',opt,'optimf',@fminlbfgs);
%     save(gp_name_id,'gp');
else
    disp('GP is already in cache directory - I am just loading it. Please delete the file if you actually wanna redo training.')
%     load(gp_name_id,'gp');
end




global training_data R_inv U Lambda

% Getting parameters, matrices and constants out of the multi class GP

% test gp 
[params_for_gp_toolbox,KWii,Kstarstar,trainedSystem,ym_predict] = get_multiclass_gp_params(gp,X_train,y_train,X_test,y_test);



    all_pred=[all_pred;ym_predict];
    all_label=[all_label;y_test];



end

% % groups for confusionmat 
all_pred_group=zeros(size(all_pred,1),1);
for i=1:size(all_pred,1)
    pred=all_pred(i,:);
    if pred(1,1)==1
      all_pred_group(i,1)=1;
    elseif  pred(1,2)==1
       all_pred_group(i,1)=2;
    else
              all_pred_group(i,1)=3;
    end
end


all_tar_group=zeros(size(all_label,1),1);
all_tar_group(1:87)=1;
all_tar_group(88:2*87)=2;
all_tar_group(2*87+1:end)=3;
% 
% 
% confusionmat(all_pred_group,all_tar_group)
% [c,cm,ind,per]=confusion(all_pred,dataset_tar)
% 
% plotconfusion(all_pred_group,all_tar_group)


            

% evaluate performance 
% shadi
% AA=confusionmat(all_pred_group,all_tar_group);
AA=confusionmat(all_tar_group,all_pred_group);

F1n=2*AA(1,1)/(sum(AA(1,:))+sum(AA(:,1)));
F1a=2*AA(2,2)/(sum(AA(2,:))+sum(AA(:,2)));
F1o=2*AA(3,3)/(sum(AA(3,:))+sum(AA(:,3)));
% F1p=2*AA(4,4)/(sum(AA(4,:))+sum(AA(:,4)));
F1=(F1a+F1o+F1n)/3;


str = ['F1 measure for pain level 1 :  ' '%1.4f\n'];
fprintf(str,F1n)
str = ['F1 measure for pain level 2:  ' '%1.4f\n'];
fprintf(str,F1a)
str = ['F1 measure for pain level 3:  ' '%1.4f\n'];
fprintf(str,F1o)
% str = ['F1 measure for Noisy recordings:  ' '%1.4f\n'];
% fprintf(str,F1p)
str = ['Final F1 measure:  ' '%1.4f\n'];
fprintf(str,F1)


missedm = (sum(sum(abs( all_pred-all_label)))/2)/size(all_label,1);
disp('Test set accuracy: ')
disp(1 - missedm)


% 
% % interpretability 
% 
% R_inv = KWii*params_for_gp_toolbox.sigma;
% R_inv = 0.5*(R_inv + R_inv');
% [U,Lambda] = eig(R_inv);
% Lambda = diag(Lambda);
% training_data = X_train;
% clear X_train
% 
% if bound_comp_opts.numberOfThreads > 1
%     if isempty(gcp('nocreate'))
%         parpool(bound_comp_opts.numberOfThreads);
%     end
% end
% 
% 
% 
%     
%   
% 
% %% Formal Bound
% 
% pi_LLs = zeros(length(Testpoints),length(n_eps) - 1,size(X_test,2));
% pi_LUs = zeros(length(Testpoints),length(n_eps) - 1,size(X_test,2));
% pi_ULs = zeros(length(Testpoints),length(n_eps) - 1,size(X_test,2));
% pi_UUs = zeros(length(Testpoints),length(n_eps) - 1,size(X_test,2));
% 
% for dd = 1%:length(Testpoints)
%     disp(['Test point: ', int2str(dd)])
%     [~,bound_comp_opts.classIdx] = max(y_test(dd,:));
%     disp(['Class idx: ', int2str(bound_comp_opts.classIdx)])
%     testIdx = Testpoints(dd);
%     testPoint = X_test(testIdx,:);
% 
%     
%     
%     for idxPix = 1:length(testPoint)
%         bound_comp_opts.pix_2_mod = [idxPix];
%         disp(['Pixel index: ', int2str(idxPix)])
%         for ii = 2:n_eps
%             bound_comp_opts.epsilon = epsilons(ii);
%             disp('Current epsilon')
%             disp(bound_comp_opts.epsilon)
%             disp('Right side interval:')
%             [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
%                 bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
%             bound_comp_opts.x_L = x_L;
%             bound_comp_opts.x_U = x_U;
%             [pi_LLs(dd,ii - 1,idxPix),pi_UUs(dd,ii - 1,idxPix),...
%                 pi_LUs(dd,ii - 1,idxPix),pi_ULs(dd,ii - 1,idxPix)]...
%                 = main_pi_hat_computation('all',testPoint,testIdx,...
%                 params_for_gp_toolbox,bound_comp_opts,trainedSystem);
%        
%         end
%     end
% end
% 
% T.pi_LLs = pi_LLs;
% T.pi_LUs = pi_LUs;
% T.pi_ULs = pi_ULs;
% T.pi_UUs = pi_UUs;

%save([dirName,'/Results.mat ','T']);