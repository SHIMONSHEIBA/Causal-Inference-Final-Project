import numpy as np
import pandas as pd

common_support_check = False

# test trim common support
if common_support_check:
    d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
    data = pd.DataFrame(data = d)

    group_min_max = (data.groupby('treatment')
                     .propensity.agg({"min_propensity": np.min, "max_propensity": np.max}))

    min_common_support = np.max(group_min_max.min_propensity)
    print('min common support:{}'.format(min_common_support))
    max_common_support = np.min(group_min_max.max_propensity)
    print('max common support:{}'.format(max_common_support))

    common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
    control = (data['treatment'] == 0)
    print('control:{}'.format(control))
    treated = (data['treatment'] == 1)
    print('treated:{}'.format(treated))
    print(data[common_support])
######################################################################
d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
data = pd.DataFrame(data = d)

# print((data.groupby("treatment")
#                          ["propensity"].agg({"min_propensity": np.min, "max_propensity": np.max})))
min_common_support = 0.2
print(data["propensity"] >= min_common_support)
#     data.propensity)
# print(data["propensity"])