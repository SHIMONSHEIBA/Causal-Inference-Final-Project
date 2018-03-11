# import numpy as np
# import pandas as pd
# from collections import defaultdict
# import pickle
# import re
# # common_support_check = False
# #
# # # test trim common support
# # if common_support_check:
# #     d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
# #     data = pd.DataFrame(data = d)
# #
# #     group_min_max = (data.groupby('treatment')
# #                      .propensity.agg({"min_propensity": np.min, "max_propensity": np.max}))
# #
# #     min_common_support = np.max(group_min_max.min_propensity)
# #     print('min common support:{}'.format(min_common_support))
# #     max_common_support = np.min(group_min_max.max_propensity)
# #     print('max common support:{}'.format(max_common_support))
# #
# #     common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
# #     control = (data['treatment'] == 0)
# #     print('control:{}'.format(control))
# #     treated = (data['treatment'] == 1)
# #     print('treated:{}'.format(treated))
# #     print(data[common_support])
# # ######################################################################
# # d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
# # data = pd.DataFrame(data = d)
# #
# # # print((data.groupby("treatment")
# # #                          ["propensity"].agg({"min_propensity": np.min, "max_propensity": np.max})))
# # min_common_support = 0.2
# # print(data["propensity"] >= min_common_support)
# # #     data.propensity)
# # # print(data["propensity"])
#
# # submissions = pd.DataFrame()
# # submissions = submissions.from_csv("C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\Causal-Inference-Final-Project\\"
# #                      "importing_change_my_view\\all submissions.csv")
# #
# # comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
# #                                                 "Causal-Inference-Final-Project\\"
# #                      "importing_change_my_view\\all submissions comments.csv", index_col=False)
# #
# # deltas = defaultdict(dict)
# # OP_deltas_comments_ids = defaultdict(list)
# # delta_tokens = ['Δ', '!delta', '∆', '&#8710;']
# # num_of_deltas = 0
# # OP_deltas = defaultdict(dict)
# # for index, row in comments.iterrows():
# #         if row.loc['comment_is_submitter'] == True and (delta_tokens[0] in row.loc['comment_body'] or
# #                                                                 delta_tokens[1] in row.loc['comment_body'] or
# #                                                                 delta_tokens[2] in row.loc['comment_body'] or
# #                                                                 delta_tokens[3] in row.loc['comment_body']) \
# #                 and len(row.loc['comment_body']) > 50:
# #             num_of_deltas +=1
# #             print(num_of_deltas)
# #             OP_deltas_comments_ids[row.submission_id].append(row.parent_id)
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_OP"] = row.comment_author
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_date"] = row.comment_created_utc
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_body"] = row.comment_body
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_path"] = row.comment_path
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_id"] = row.comment_id
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_parent_id"] = row.parent_id
# #             OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_submission_id"] = row.submission_id
# #
#
# pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
#
# OP_deltas_comments_ids = pickle.load(pkl_file)
# print("data")
#
# deltas = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                                 "Causal-Inference-Final-Project\\"
#                      "importing_change_my_view\\all deltas - Copy.csv", index_col=False)
#
# OP_deltas_comments_ids_deltalog = defaultdict(list)
# comments_with_delta_ids = list()
# delta_count = 0
# for df_index, row in deltas.iterrows():
#
#     # obtain submissionid
#     submission_id_indexes = [m.start() for m in re.finditer('/comments/', deltas.loc[df_index, "delta_selftext"])]
#     if len(submission_id_indexes) < 1:
#         print("bug")
#     submission_id_text = deltas.loc[df_index, "delta_selftext"][:submission_id_indexes[0]+20]
#     submission_id_text_parsed = submission_id_text.split("/")
#     comments_idx = submission_id_text_parsed.index("comments")
#     submission_id = submission_id_text_parsed[comments_idx+1]
#
#     # obtain OP user name
#     op_username_indexes = [m.start() for m in re.finditer('"opUsername":', deltas.loc[df_index, "delta_selftext"])]
#     op_username_text = deltas.loc[df_index, "delta_selftext"][op_username_indexes[0]:]
#     left_op_username_text = op_username_text.partition("\\n")[0]
#     op_username = left_op_username_text.partition('",\\n')[0]
#     op_username = op_username.split(": ")[1]
#     op_username = op_username.strip(",'")
#
#     # parse from delta text all comments id's that got deltas.
#     deltas_indexes = [m.start() for m in re.finditer('"awardedLink":' , deltas.loc[df_index,"delta_selftext"])]
#     delta_awarding_indexes = [m.start() for m in re.finditer('"awardingUsername":' , deltas.loc[df_index,"delta_selftext"])]
#
#     if len(deltas_indexes) != len(delta_awarding_indexes):
#         print("match error")
#         break
#
#     for delta_list_idx, delta_index in enumerate(deltas_indexes):
#
#         #get & check if awarding is OP
#         awarding_text = deltas.loc[df_index, "delta_selftext"][delta_awarding_indexes[delta_list_idx]:]
#         left_awarding_text = awarding_text.partition("\\n")[0]
#         awarding_username = left_awarding_text.partition('",\\n')[0]
#         awarding_username = awarding_username.split(": ")[1]
#         awarding_username = awarding_username.strip(",'")
#
#         if awarding_username != op_username:
#             #print("not OP's delta")
#             continue
#
#         # get awarded delta comment id index
#         text = deltas.loc[df_index,"delta_selftext"][delta_index:]
#         left_text = text.partition("\\n")[0]
#         splited = text.split('/')
#         for idx, parts in enumerate(splited):
#             if "cmv_" in parts:
#                 comment_id = splited[idx + 1].partition('",\\n')[0]
#                 print(comment_id)
#                 delta_count += 1
#                 print(delta_count)
#                 comments_with_delta_ids.append(comment_id)
#                 OP_deltas_comments_ids_deltalog[submission_id].append(comment_id)
#
#
# comments_with_delta_ids = list(set(comments_with_delta_ids))
# print("num of deltas from deltalog is: {}".format(len(comments_with_delta_ids)))
#
#
# comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                           "Causal-Inference-Final-Project\\"
#                                           "importing_change_my_view\\all submissions comments - Copy.csv", index_col=False)
# comments["delta"] = 0
#
# for index, row in comments.iterrows():
#
#     if type(row.loc['comment_id']) is float:
#         #TODO: VALIDATE WHY SO MUCH NAN
#         print("not real comment id : {}".format(row.loc['comment_id']))
#         continue
#
#     deltalog_comments = list()
#     manual_comments = list()
#
#     # get delta comments of this submissionid
#     try:
#         deltalog_comments = OP_deltas_comments_ids_deltalog[row.loc['submission_id']]
#     except ValueError:
#         print("no delta comments for submission: {} in deltalog".format(row.loc['submission_id']))
#
#     try:
#         manual_comments = OP_deltas_comments_ids[row.loc['submission_id']]
#         manual_comments = [w.lstrip("b") for w in manual_comments]
#         manual_comments = [w.strip("'") for w in manual_comments]
#         manual_comments = [w.lstrip("t1_") for w in manual_comments]
#
#     except ValueError:
#         print("no delta comments for submission: {} in manual".format(row.loc['submission_id']))
#
#     # check if this comment got delta
#     if row.loc['comment_id'].lstrip("b").strip("'") in deltalog_comments \
#             or row.loc['comment_id'].lstrip("b").strip("'") in manual_comments:
#         comments.loc[index, "delta"] = 1
#
# print("VALUE COUNT:")
# print(comments['delta'].value_counts())
#
# count_deltas = list()
# for key, value in OP_deltas_comments_ids_deltalog.items():
#     count_deltas += value
# print("num of deltas in manual: {}".format(len(count_deltas)))
# for k, value in OP_deltas_comments_ids.items():
#     count_deltas += value
# count_deltas = list(set(count_deltas))
# print("TOTAL {} deltas".format(len(count_deltas)))
#
# comments.to_csv(path_or_buf="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                             "Causal-Inference-Final-Project\\"
#                             "importing_change_my_view\\all submissions comments with label.csv")


# import pandas as pd
# import os
# base_directory = os.path.abspath(os.curdir)
# change_my_view_directory = os.path.join(base_directory, 'change my view')
# data = pd.read_csv(os.path.join(change_my_view_directory, 'filter_comments_submissions.csv'))
#
# reut = 1

myString = 'Position of a character'
if 'Position of a character' in myString:
    print('yes')