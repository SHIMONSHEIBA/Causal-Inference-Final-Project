import pickle
import pandas as pd

# pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
# OP_deltas_comments_ids = pickle.load(pkl_file)

pkl_file = open('OP_deltas_comments_ids_deltalog.pickle', 'rb')
OP_deltas_comments_ids_deltalog = pickle.load(pkl_file)

pkl_file = open('OP_deltas_comments_ids_of_submissions_not_in_data.pickle', 'rb')
OP_deltas_comments_ids_deltalog_not_in_data = pickle.load(pkl_file)


# all_submissions_comments_label = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                                           "Causal-Inference-Final-Project\\"
#                                                           "importing_change_my_view\\all submissions comments with label of submission delta log not in data.csv",index_col=False)

comment_ids = list()
for key, value in OP_deltas_comments_ids_deltalog_not_in_data.items():
    comment_ids = comment_ids + value

comment_ids_deltalog = list()
for key, value in OP_deltas_comments_ids_deltalog.items():
    comment_ids_deltalog = comment_ids_deltalog + value

# pkl_file = open('OP_deltas_comments_ids_latest.pickle', 'rb')
# OP_deltas_comments_ids_latest = pickle.load(pkl_file)
#
# comment_ids_latest = list()
# for key, value in OP_deltas_comments_ids_latest.items():
#     comment_ids_latest = comment_ids_latest + value
#
#
# deltas_log_intersection_keys = set(all_submissions_comments_label["submission_id"].str.lstrip("b").str.strip("'")).intersection(set(OP_deltas_comments_ids_deltalog.keys()))
# log_keys_not_in_data = list(set(OP_deltas_comments_ids_deltalog.keys()).difference(deltas_log_intersection_keys))
#
# with open('submissions_from_delta_log_not_in_data.pickle', 'wb') as handle:
#     pickle.dump(log_keys_not_in_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# comment_log_deltas_not_in_data = list()
# for items in log_keys_not_in_data:
#     comment_log_deltas_not_in_data = comment_log_deltas_not_in_data + OP_deltas_comments_ids_deltalog[items]
print("shimon")