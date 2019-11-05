function bound_comp_opts = default_parameter_checkGP_OAT(varargin)
bound_comp_opts.numberOfThreads  = 1;
bound_comp_opts.tollerance = 0.02; %bound tollerance required
bound_comp_opts.epsilons_vec = [0.2];
bound_comp_opts.points_to_analyse_vec = [1];
bound_comp_opts.mod_modus = 'OAT';
bound_comp_opts.pix_2_mod = 'all';
bound_comp_opts.constrain_2_one = false;
bound_comp_opts.max_iterations = 10000;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-20;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.likmode = 'analytical';
bound_comp_opts.mode = 'binarypi';

for ii = 1:2:length(varargin)
    field = varargin{1};
    value = varargin{2};
    if isefield(bound_comp_opts,field)
        bound_comp_opts.(field) = value;
    else
        warning('Option requested does not exist. I am ignoring the entry...')
    end
end

end