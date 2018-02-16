# from scipy.stats import lognorm
#
#
# f = 0.00004
#
# rv = lognorm(f)
#
# print(rv)
#
#
#
# key_min = min(efficient_language_model_object.relevant_referrals_1gram_prob_calc.keys(), key=(lambda k: efficient_language_model_object.relevant_referrals_1gram_prob_calc[k]))
# print(efficient_language_model_object.relevant_referrals_1gram_prob_calc[key_min])
#
# -6500.0
#
# key_min = min(efficient_language_model_object.other_ref_text_1gram_prob_calc.keys(), key=(lambda k: efficient_language_model_object.other_ref_text_1gram_prob_calc[k]))
# print(efficient_language_model_object.other_ref_text_1gram_prob_calc[key_min])
#
# -6200.0
#
# key_min = min(non_efficient_language_model_object.relevant_referrals_1gram_prob_calc.keys(), key=(lambda k: non_efficient_language_model_object.relevant_referrals_1gram_prob_calc[k]))
# print(non_efficient_language_model_object.relevant_referrals_1gram_prob_calc[key_min])
#
# -7800.0
#
# key_min = min(non_efficient_language_model_object.other_ref_text_1gram_prob_calc.keys(), key=(lambda k: non_efficient_language_model_object.other_ref_text_1gram_prob_calc[k]))
# print(non_efficient_language_model_object.other_ref_text_1gram_prob_calc[key_min])
#
# -6100.0

import pandas as pd

# word_prob_vc = pd.DataFrame.from_records([{'pos': 0.8, 'neg': 0,
#                                            'obj': 0.9}])
#
# f = lambda x: True if x  > 0.8 else False
# pos_neg_rules_vec = pd.DataFrame.from_records([{'pos': f(word_prob_vc.iloc[0,2]),
#                                                         'neg': f(word_prob_vc.iloc[0,0]),
#                                                         'obj': f(word_prob_vc.iloc[0,1])}])
#
# print(pos_neg_rules_vec)

g = lambda x,y: True if x > y else False

print(g(0.3,0.3))