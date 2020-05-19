feat_groups = {(1:5), [6,7], [8] };

for i_feat_groups = 1:length(feat_groups)
    clearvars -except feat_groups i_feat_groups
    feat_group = feat_groups{i_feat_groups};
    gp_training_deap_partial_feats;
    save(['group_',int2str(feat_group),'_results.csv'],[res;F_score])
end

