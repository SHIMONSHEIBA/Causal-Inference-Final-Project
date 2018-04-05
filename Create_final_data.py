import pandas as pd
import os
from functools import reduce
from copy import copy


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")


# Create delta + comment_depth:
delta_depth = pd.read_csv(os.path.join(data_directory, 'units_0304.csv'))
delta_depth = delta_depth[['comment_id', 'delta', 'comment_depth']]
print('delta_depth size: ', delta_depth.shape)

# create features DF
features = pd.read_csv(os.path.join(data_directory, 'features_CMV.csv'))
# choose features to use:
features = features[['comment_id', 'commenter_number_submission', 'submitter_number_submission',
                     'submitter_seniority_days', 'is_first_comment_in_tree',
                     'commenter_number_comment', 'submitter_number_comment',
                     'number_of_comments_in_tree_by_comment_user', 'commenter_seniority_days',
                     'time_ratio', 'comment_len', 'submission_len', 'title_len', 'time_between_messages',
                     'time_until_first_comment', 'time_between_comment_first_comment', 'time_between']]
time_between_0 = features.loc[features['time_between'] == 0]
print(time_between_0.shape)
print('features size: ', features.shape)
features.loc[features['time_between_messages'] == 0, 'time_ratio'] = 0
print(features[features.isnull().any(axis=1)]['time_ratio'])

# create fixed features DF
fixed_features = pd.read_csv(os.path.join(data_directory, 'fixed_features_new.csv'))
# choose features to use:
fixed_features = fixed_features[['number_of_comments_in_tree_from_submitter', 'number_of_respond_by_submitter_total',
                                 'respond_to_comment_user_responses_ratio', 'number_of_respond_by_submitter',
                                 'respond_to_comment_user_all_ratio', 'respond_total_ratio', 'comment_id']]
print('fixed_features size: ', fixed_features.shape)

# create treatment DF
treatment = pd.read_csv(os.path.join(data_directory, 'data_label_treatment_all_0304.csv'))
treatment = treatment[['comment_id', 'treated']]
print('treatment size: ', treatment.shape)
added_treatment = pd.read_csv(os.path.join(data_directory, 'data_label_treatment_all_no_parent.csv'))
added_treatment = added_treatment[['comment_id', 'treated']]
print('added_treatment size: ', added_treatment.shape)
treatment = pd.concat([treatment, added_treatment], axis=0)
print('all treatment size: ', treatment.shape)

# create text features (sentiment and percent of adj) DF
text_features = pd.read_csv(os.path.join(data_directory, 'sentiment_analysis_CMV.csv'))
text_features = text_features[['comment_id', 'nltk_com_sen_pos', 'nltk_com_sen_neg', 'nltk_com_sen_neutral',
                               'nltk_sub_sen_pos', 'nltk_sub_sen_neg', 'nltk_sub_sen_neutral', 'nltk_title_sen_pos',
                               'nltk_title_sen_neg', 'nltk_title_sen_neutral', 'nltk_sim_sen', 'percent_adj']]
print('text_features size: ', text_features.shape)

# create topic model DF
topic_model = pd.read_csv(os.path.join(data_directory, 'topic_model_CMV.csv'))
print('topic_model size: ', topic_model.shape)

# create similarity feature
similarity = pd.read_csv(os.path.join(data_directory, 'units_with_sim_tfidf_cos.csv'))
print('similarity size: ', similarity.shape)

# features_comment_id = features['comment_id']
# topic_model_comment_id = topic_model['comment_id']
# in_topic = set(topic_model_comment_id) - set(features_comment_id)
# in_features = set(features_comment_id) - set(topic_model_comment_id)

# join all the DFs
dfs = [treatment, text_features, topic_model, similarity, delta_depth, fixed_features, features]
for df in dfs:
    df['comment_id'] = df.comment_id.str.lstrip("b'")
    df['comment_id'] = df.comment_id.str.rstrip("'")

features_comment_id = fixed_features['comment_id']
topic_model_comment_id = topic_model['comment_id']
treatment_comment_id = treatment['comment_id']

in_topic_treatment = set(treatment_comment_id) - set(topic_model_comment_id)
in_topic = set(topic_model_comment_id) - set(features_comment_id)
in_features = set(features_comment_id) - set(topic_model_comment_id)

features_ok = set(features_comment_id) - set(in_features)

fix = copy(fixed_features)
fix_ok = fix.loc[fix['comment_id'].isin(list(features_ok))]
fix_to_fix = fix.loc[fix['comment_id'].isin(list(in_features))]
fix_to_fix['comment_id'] = fix_to_fix['comment_id'] + 'b'
fix = pd.concat([fix_ok, fix_to_fix], axis=0)
print('fix shape:', fix.shape)

treatment_ok = set(treatment_comment_id) - set(in_topic_treatment)
treatment_fix = copy(treatment)
treatment_fix_ok = treatment_fix.loc[treatment_fix['comment_id'].isin(list(treatment_ok))]
treatment_fix_to_fix = treatment_fix.loc[treatment_fix['comment_id'].isin(list(in_topic_treatment))]
treatment_fix_to_fix['comment_id'] = treatment_fix_to_fix['comment_id'] + 'b'
treatment_fix = pd.concat([treatment_fix_ok, treatment_fix_to_fix], axis=0)
print('treatment_fix shape:', treatment_fix.shape)

dfs = [treatment_fix, text_features, topic_model, similarity, delta_depth, fix, features]
df_final = reduce(lambda left, right: pd.merge(left, right, on='comment_id'), dfs)
#
# check = pd.merge(treatment_fix, text_features, on='comment_id')
# print('treatment, text_features size: ', check.shape)
# check = pd.merge(treatment_fix, topic_model, on='comment_id')
# print('treatment, topic_model size: ', check.shape)
# check = pd.merge(treatment_fix, similarity, on='comment_id')
# print('treatment, similarity size: ', check.shape)
# check = pd.merge(treatment_fix, delta_depth, on='comment_id')
# print('treatment, delta_depth size: ', check.shape)
# check = pd.merge(treatment_fix, fix, on='comment_id')
# print('treatment, fixed_features size: ', check.shape)
# check = pd.merge(treatment_fix, features, on='comment_id')
# print('treatment, features size: ', check.shape)
# check = pd.merge(text_features, topic_model, on='comment_id')
# print('text_features, topic_model size: ', check.shape)
# check = pd.merge(text_features, similarity, on='comment_id')
# print('text_features, similarity size: ', check.shape)
# check = pd.merge(text_features, delta_depth, on='comment_id')
# print('text_features, delta_depth size: ', check.shape)
# check = pd.merge(text_features, fix, on='comment_id')
# print('text_features, fixed_features size: ', check.shape)
# check = pd.merge(text_features, features, on='comment_id')
# print('text_features, topic_model size: ', check.shape)
# check = pd.merge(topic_model, similarity, on='comment_id')
# print('topic_model, similarity size: ', check.shape)
# check = pd.merge(topic_model, delta_depth, on='comment_id')
# print('topic_model, delta_depth size: ', check.shape)
# check = pd.merge(topic_model, fix, on='comment_id')
# print('topic_model, fixed_features size: ', check.shape)
# check = pd.merge(topic_model, features, on='comment_id')
# print('topic_model, features size: ', check.shape)
# check = pd.merge(similarity, delta_depth, on='comment_id')
# print('similarity, delta_depth size: ', check.shape)
# check = pd.merge(similarity, features, on='comment_id')
# print('similarity, features size: ', check.shape)
# check = pd.merge(similarity, fix, on='comment_id')
# print('treatment, fixed_features size: ', check.shape)
# check = pd.merge(delta_depth, features, on='comment_id')
# print('delta_depth, features size: ', check.shape)
# check = pd.merge(delta_depth, fix, on='comment_id')
# print('delta_depth, fixed_features size: ', check.shape)
# check = pd.merge(features, fix, on='comment_id')
# print('features, fixed_features size: ', check.shape)

del df_final['Unnamed: 0_x']
del df_final['Unnamed: 0_y']

print('Final vector length including delta and comment_id is: ', df_final.shape)

df_final.to_csv(os.path.join(data_directory, 'final_df_CMV_after_fix.csv'), encoding='utf-8')

treatment = df_final.loc[df_final['treated'] == 1]
print('size of treated = {}'.format(treatment.shape))
control = df_final.loc[df_final['treated'] == 0]
print('size of control = {}'.format(control.shape))
no_parent = df_final.loc[df_final['treated'] == -1]
print('size of no_parent = {}'.format(no_parent.shape))

treatment_delta = treatment.loc[treatment['delta'] == 1]
print('size of treated_delta = {}'.format(treatment_delta.shape))
treatment_no_delta = treatment.loc[treatment['delta'] == 0]
print('size of treatment_no_delta = {}'.format(treatment_no_delta.shape))
control_delta = control.loc[control['delta'] == 1]
print('size of control_delta = {}'.format(control_delta.shape))
control_no_delta = control.loc[control['delta'] == 0]
print('size of control_no_delta = {}'.format(control_no_delta.shape))
