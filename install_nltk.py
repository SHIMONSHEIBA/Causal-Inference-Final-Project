import pandas as pd
import pickle

all_data0304 = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                          "Causal-Inference-Final-Project\\importing_change_my_view\\"
                                                          "all_data_0304.csv",
                                       index_col=False)




new_delta = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                          "Causal-Inference-Final-Project\\importing_change_my_view\\"
                                                          "all submissions comments with label.csv",
                                       index_col=False)

pkl_file = open('OP_deltas_comments_ids_deltalog.pickle', 'rb')
OP_deltas_comments_ids_deltalog = pickle.load(pkl_file)

# TODO: change save/load places to self class variables
pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
OP_deltas_comments_ids = pickle.load(pkl_file)

pkl_file = open('OP_deltas_comments_ids_latest.pickle', 'rb')
OP_deltas_comments_ids_latest = pickle.load(pkl_file)

delta_comments_latest = list()
delta_comments_new = list()
delta_comments_log = list()

for sub, com_list in OP_deltas_comments_ids_latest.items():
    delta_comments_latest += com_list

delta_comments_latest_set = list(set(delta_comments_latest))
delta_comments_latest_len = len(delta_comments_latest_set)
print("num of latest is ", delta_comments_latest_len)

for sub, com_list in OP_deltas_comments_ids.items():
    delta_comments_new += com_list

delta_comments_new_set = list(set(delta_comments_new))
delta_comments_new_len = len(delta_comments_new_set)
print("num of new is ", delta_comments_new_len)

for sub, com_list in OP_deltas_comments_ids_deltalog.items():
    delta_comments_log += com_list

delta_comments_log_set = list(set(delta_comments_log))
delta_comments_log_len = len(delta_comments_log_set)
print("num of log is ", delta_comments_log_len)

removed = list(set(delta_comments_latest_set) - set(delta_comments_new_set))
print(removed)
print("hello")