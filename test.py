import numpy as np
import pandas as pd
from collections import defaultdict
# common_support_check = False
#
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

submissions = pd.DataFrame()
submissions = submissions.from_csv("C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\Causal-Inference-Final-Project\\"
                     "importing_change_my_view\\all submissions.csv")

comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                "Causal-Inference-Final-Project\\"
                     "importing_change_my_view\\all submissions comments.csv", index_col=False)

deltas = defaultdict(dict)
token1 = 'Δ'
token2 = '!delta'
token3 = '∆'
token4 = '&#8710;'
num_of_deltas = 0
for index, row in comments.iterrows():
        if row.loc['comment_is_submitter'] == True and (token1 in row.loc['comment_body'] or
                                                                token2 in row.loc['comment_body'] or
                                                                token3 in row.loc['comment_body'] or
                                                                token4 in row.loc['comment_body']) \
                and len(row.loc['comment_body']) > 50:
            num_of_deltas +=1
            print(num_of_deltas)




