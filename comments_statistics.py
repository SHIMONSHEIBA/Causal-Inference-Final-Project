import pandas as pd
import os

base_directory = os.path.abspath(os.curdir)
change_my_view_directory = os.path.join(base_directory, 'change my view')
comments = pd.read_csv(os.path.join(change_my_view_directory, 'all submissions comments with label.csv'))
submissions = pd.read_csv(os.path.join(change_my_view_directory, 'all submissions.csv'))

comments['submission_id'] = comments.submission_id.str.slice(2, -1)
submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")

join_result = comments.merge(submissions_drop, on='submission_id', how='inner')
join_result['comment_len'] = join_result['comment_body'].str.len()
join_result['time_between'] =\
    join_result['comment_created_utc'].astype('float') - join_result['submission_created_utc'].astype('float')
join_result['comment_depth'] = join_result['comment_depth'].astype('int')

group_depth = join_result.groupby('comment_depth')['comment_id']
group_len = join_result.groupby('comment_len')['comment_id']
group_time = join_result.groupby('time_between')['comment_id']
group_submission_num_comments = join_result.groupby('submission_num_comments')['comment_id']

# join_result.to_csv(os.path.join(change_my_view_directory, 'all_comments_submissions.csv'))
# join_result.to_pickle(os.path.join(change_my_view_directory, 'all_comments_submissions.pkl'))

group_depth_count = group_depth.count()
group_len_count = group_len.count()
group_time_count = group_time.count()
group_submission_num_comments_count = group_submission_num_comments.count()

# print(group_submission_num_comments_count)

filter_results = join_result.loc[(join_result['time_between'] < 10080.0) & (join_result['comment_len'] > 150)
                                 & (join_result['comment_author'] != 'DeltaBot')
                                 & (~join_result['comment_body'].str.contains('your comment has been removed:'))
                                 & (join_result['comment_body'].str.contains('[deleted]'))]

units = filter_results.loc[(filter_results['comment_author'] != filter_results['submission_author'])]

filter_results.to_csv(os.path.join(change_my_view_directory, 'all_data.csv'))
filter_results.to_pickle(os.path.join(change_my_view_directory, 'all_data.pkl'))

units.to_csv(os.path.join(change_my_view_directory, 'units.csv'))
units.to_pickle(os.path.join(change_my_view_directory, 'units.pkl'))

# print(filter_results.shape)
# delta = filter_results.loc[filter_results['delta'] == 1]
# print(delta.shape)
# after_17 = filter_results.loc[filter_results['comment_created_utc'] >= 1475280000]
# print(after_17.shape)
# delta_after_17 = after_17.loc[filter_results['delta'] == 1]
# print(delta_after_17.shape)
# no_delta_after_17 = after_17.loc[filter_results['delta'] == 0]
# print(no_delta_after_17.shape)

# sub_delta = delta['submission_id']
# sub_delta = sub_delta.drop_duplicates()
# sub_delta = list(sub_delta)
# after_17 = filter_results.loc[(filter_results['comment_created_utc'] >= 1483228800) & (filter_results['delta'] == 0)]
# before_17_no_delta_from_sub_delta = filter_results.loc[(filter_results['submission_id'].isin(sub_delta)) &
#                                                        (filter_results['delta'] == 0) &
#                                                        (filter_results['comment_created_utc'] < 1483228800)]
# final_results = pd.concat([delta, after_17, before_17_no_delta_from_sub_delta])
#
# final_results.to_csv(os.path.join(change_my_view_directory, 'filter_comments_submissions_new_filter.csv'))
# final_results.to_pickle(os.path.join(change_my_view_directory, 'filter_comments_submissions_new_filter.pkl'))
