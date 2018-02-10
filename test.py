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
first = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/096260 - causality/Causal-Inference-Final-Project/Causal-Inference-Final-Project/final data to use/predictedResultsEfficiency_pass_threshold_3403_comments.xlsx')
second = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/096260 - causality/Causal-Inference-Final-Project/Causal-Inference-Final-Project/final data to use/predictedResultsEfficiency_all_data_final_1.xlsx')
third = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/096260 - causality/Causal-Inference-Final-Project/Causal-Inference-Final-Project/final data to use/predictedResultsEfficiency_all_data.xlsx')

final = pd.concat([first, second, third], axis=0)

print(final.shape)

# final.to_csv('/Users/reutapel/Documents/Technion/Msc/096260 - causality/Causal-Inference-Final-Project/Causal-Inference-Final-Project/final data to use/predictedResultsEfficiency_final.csv')

shimon = pd.read_excel('/Users/reutapel/Documents/Technion/Msc/096260 - causality/Causal-Inference-Final-Project/Causal-Inference-Final-Project/FinalFeatures_with_comment_time_shimon.xlsx', encoding='utf-8')
reut = 1