import pandas as pd
import os


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")


# create features DF
features = pd.read_csv(os.path.join(data_directory, 'features_CMV.csv'))
# choose features to use:
features = features[['comment_id', 'commenter_number_submission', 'submitter_number_submission',
                     'number_of_comments_in_tree_from_submitter', 'number_of_respond_by_submitter_total',
                     'respond_to_comment_user_responses_ratio', 'submitter_seniority_days', 'is_first_comment_in_tree',
                     'commenter_number_comment', 'submitter_number_comment', 'comment_depth',
                     'number_of_comments_in_tree_by_comment_user', 'number_of_respond_by_submitter',
                     'respond_to_comment_user_all_ratio', 'respond_total_ratio', 'commenter_seniority_days',
                     'time_ratio', 'comment_len', 'submission_len', 'title_len', 'time_between_messages',
                     'time_until_first_comment', 'time_between_comment_first_comment']]

# create treatment DF
treatment = pd.read_csv(os.path.join(data_directory, 'data_label_treatment_all_0304.csv'))
treatment = treatment[['comment_id', 'treated']]

# create text features (sentiment and percent of adj) DF
text_features = pd.read_csv(os.path.join(data_directory, 'sentiment_analysis_CMV.csv'))
text_features = text_features[['comment_id', 'nltk_com_sen_pos', 'nltk_com_sen_neg', 'nltk_com_sen_neutral',
                               'nltk_sub_sen_pos', 'nltk_sub_sen_neg', 'nltk_sub_sen_neutral', 'nltk_title_sen_pos',
                               'nltk_title_sen_neg', 'nltk_title_sen_neutral', 'nltk_sim_sen', 'percent_adj']]

# create topic model DF
topic_model = pd.read_csv(os.path.join(data_directory, 'sentiment_analysis_CMV.csv'))