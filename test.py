import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import re
# common_support_check = False
import csv
import os
import CreateFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from optparse import OptionParser
import sys
from sklearn.metrics.pairwise import cosine_similarity

#############################################test matching################################
# # test trim common support
# if common_support_check:
#     d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
#     data = pd.DataFrame(data = d)
#
#     group_min_max = (data.groupby('treatment')
#                      .propensity.agg({"min_propensity": np.min, "max_propensity": np.max}))
#
#     min_common_support = np.max(group_min_max.min_propensity)
#     print('min common support:{}'.format(min_common_support))
#     max_common_support = np.min(group_min_max.max_propensity)
#     print('max common support:{}'.format(max_common_support))
#
#     common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
#     control = (data['treatment'] == 0)
#     print('control:{}'.format(control))
#     treated = (data['treatment'] == 1)
#     print('treated:{}'.format(treated))
#     print(data[common_support])
# ######################################################################
# d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
# data = pd.DataFrame(data = d)
#
# # print((data.groupby("treatment")
# #                          ["propensity"].agg({"min_propensity": np.min, "max_propensity": np.max})))
# min_common_support = 0.2
# print(data["propensity"] >= min_common_support)
# #     data.propensity)
# # print(data["propensity"])

########################################################################

###################################validate deltas##########################
test_deltas = False

if test_deltas:
    comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                              "Causal-Inference-Final-Project\\"
                                              "importing_change_my_view\\all submissions comments with label.csv", index_col=False)

    pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
    OP_deltas_comments_ids = pickle.load(pkl_file)

    pkl_file = open('OP_deltas_comments_ids_deltalog.pickle', 'rb')
    OP_deltas_comments_ids_deltalog = pickle.load(pkl_file)

    # see label distribution in data
    print("VALUE COUNT:")
    print(comments['delta'].value_counts())

    # validate that total number of deltas equal to the label distribution above
    deltalog_manual_values_comment_list = list()
    for key, value in OP_deltas_comments_ids_deltalog.items():
        deltalog_manual_values_comment_list += value
    deltalog_manual_values_comment_list = set(deltalog_manual_values_comment_list)
    deltalog_manual_values_comment_list = list(deltalog_manual_values_comment_list)
    print("num of deltas in deltalog: {}".format(len(deltalog_manual_values_comment_list)))

    manual_deltas = list()
    for k, value in OP_deltas_comments_ids.items():
        value_stripped = [w.lstrip("b") for w in value]
        value_stripped = [w.lstrip("'") for w in value]
        manual_deltas += value_stripped
    manual_deltas = set(manual_deltas)
    manual_deltas = list(manual_deltas)
    print("num of deltas in manual deltas: {}".format(len(manual_deltas)))

    deltalog_manual_values_comment_list += manual_deltas
    deltalog_manual_values_comment_list = list(set(deltalog_manual_values_comment_list))
    print("TOTAL {} deltas".format(len(deltalog_manual_values_comment_list)))

    #TODO: debug 2690 deltas in deltalog and not in data
    deltalog_manual_values_comment_list = [w.lstrip("b") for w in deltalog_manual_values_comment_list]
    deltalog_manual_values_comment_list = [w.strip("'") for w in deltalog_manual_values_comment_list]
    deltalog_manual_values_comment_list = [w.lstrip("t1_") for w in deltalog_manual_values_comment_list]
    manual_deltas = [w.lstrip("b") for w in manual_deltas]
    manual_deltas = [w.strip("'") for w in manual_deltas]
    manual_deltas = [w.lstrip("t1_") for w in manual_deltas]
    manual_deltas = set(manual_deltas)
    manual_deltas = list(manual_deltas)
    deltalog_manual_values_comment_list = set(deltalog_manual_values_comment_list)
    deltalog_manual_values_comment_list = list(deltalog_manual_values_comment_list)
    deltas_from_labeled_data = comments[comments['delta'] == 1]["comment_id"]
    deltas_from_labeled_data = [w.lstrip("b") for w in deltas_from_labeled_data]
    deltas_from_labeled_data = [w.strip("'") for w in deltas_from_labeled_data]
    deltas_from_labeled_data = [w.lstrip("t1_") for w in deltas_from_labeled_data]
    both_lists = list(set(deltalog_manual_values_comment_list) | set(manual_deltas))
    not_found = list(set(both_lists) - set(deltas_from_labeled_data))
    print(comments[comments['comment_id'].str.contains('dpo25t0')]["comment_id"])

##############################################################################

##########################similarity feature######################################

test_similarity = True


if test_similarity:

    base_directory = os.path.abspath(os.curdir)
    results_directory = os.path.join(base_directory, 'importing_change_my_view')
    comments = pd.read_csv(os.path.join(results_directory, 'all submissions comments with label.csv'))
    submissions = pd.read_csv(os.path.join(results_directory, 'all submissions.csv'))

    comments['submission_id'] = comments.submission_id.str.slice(2, -1)
    submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")

    join_result = comments.merge(submissions_drop, on='submission_id', how='inner')

    join_result["submmiter_commenter_tfidf_cos_sim"] = 0


    def concat_df_rows(comment_created_utc, author, is_submission=False):
        if is_submission:
            text = join_result.loc[(join_result['comment_created_utc'] < comment_created_utc) &
                                   (join_result['comment_author'] == author)].iloc[0][["submission_title",
                                                                                       "submission_body"]]
            text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
            text["submission_body"] = text["submission_body"].partition(
                "Hello, users of CMV! This is a footnote from your moderators")[0]
            text["submission_title_and_body"] = text["submission_title"] + text["submission_body"]
            text_cat = text.str.cat(sep=' ')
            return text_cat

        text = join_result.loc[(join_result['comment_created_utc'] < comment_created_utc) &
                               (join_result['comment_author'] == author)]["comment_body"]
        text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
        text_cat = text.str.cat(sep=' ')

        return text_cat


    for index, row in join_result.iterrows():
        # define thresholds anf filter variables
        comment_created_utc = row.loc['comment_created_utc']
        comment_author = row.loc['comment_author']
        submission_author =  row.loc['submission_author']

        # all text of commenter until comment time
        text_commenter = concat_df_rows(comment_created_utc, comment_author)

        # all text of submissioner until comment time
        text_submissioner = concat_df_rows(comment_created_utc, submission_author)
        text_submissioner_submission = concat_df_rows(comment_created_utc, submission_author, True)
        text_submissioner += text_submissioner_submission

        text = [text_submissioner, text_commenter]
        tfidf = TfidfVectorizer(stop_words = 'english', lowercase = True, analyzer  = 'word', norm = 'l2',
                                smooth_idf = True, sublinear_tf  = False, use_idf  = True).fit_transform(text)
        similarity = cosine_similarity(tfidf[0:], tfidf[1:])

        join_result["submmiter_commenter_tfidf_cos_sim"].loc[index, "submmiter_commenter_tfidf_cos_sim"] = \
            similarity[0][0]

        if similarity[0][0] > 0.9 or similarity[0][0] < 0.2:
            print(similarity[0][0])
            print("text submissioner:")
            print(text_submissioner)
            print("text commenter:")
            print(text_commenter)

##################################################################################################
