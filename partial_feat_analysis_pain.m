feat_groups = {(1:5), [6,7], [8] };

for i_feat_groups = 1:length(feat_groups)
    clearvars -except feat_groups i_feat_groups
    feat_group = feat_groups{i_feat_groups};
    gp_training_pain_new_partial_feats;
    csvwrite(['group_',int2str(feat_group),'_pain_results.csv'],[res;F_score])
end

