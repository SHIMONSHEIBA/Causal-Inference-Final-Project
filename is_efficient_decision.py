import pandas as pd


all_data = pd.read_csv('is_post_before_df_total_all.csv')
resultsPredicted = pd.read_excel('FinalFeatures_with_comment_time.xlsx')

utc_time_result = all_data['referral_utc']
utc_time_result.unique()
utc_time_result.columns = ['comment_created_time']
utc_time_orig = resultsPredicted['comment_created_time']

join = utc_time_result.align(utc_time_orig)

count_before = all_data.groupby(['sub_com_index', 'referral_utc']).size()
print(count_before)

