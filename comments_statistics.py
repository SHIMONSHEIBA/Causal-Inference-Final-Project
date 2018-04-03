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

# group_depth = join_result.groupby('comment_depth')['comment_id']
# group_len = join_result.groupby('comment_len')['comment_id']
# group_time = join_result.groupby('time_between')['comment_id']
# group_submission_num_comments = join_result.groupby('submission_num_comments')['comment_id']
#
# # join_result.to_csv(os.path.join(change_my_view_directory, 'all_comments_submissions.csv'))
# # join_result.to_pickle(os.path.join(change_my_view_directory, 'all_comments_submissions.pkl'))
#
# group_depth_count = group_depth.count()
# group_len_count = group_len.count()
# group_time_count = group_time.count()
# group_submission_num_comments_count = group_submission_num_comments.count()

# print(group_submission_num_comments_count)

# filter_results = join_result.loc[(join_result['time_between'] < 10080.0) & (join_result['comment_len'] > 150)
#                                  & (join_result['comment_author'] != 'DeltaBot')
#                                  & (~join_result['comment_body'].str.contains('your comment has been removed:'))
#                                  & (join_result['comment_body'].str.contains('[deleted]'))
#                                  & (~join_result['comment_author'].isnull())
#                                  & (~join_result['submission_author'].isnull())]


filter_results = join_result.loc[(join_result['comment_len'] > 200) & (join_result['time_between'] < 604800.0)
                                 & (join_result['comment_author'] != 'DeltaBot')
                                 & (~join_result['comment_body'].str.contains('your comment has been removed:'))
                                 & (join_result['comment_body'].str.contains('[deleted]'))
                                 & (~join_result['comment_author'].isnull())
                                 & (~join_result['submission_author'].isnull())]
print('200 char and 1 week')

# filter_results.to_csv(os.path.join(change_my_view_directory, 'all_data.csv'))
# filter_results.to_pickle(os.path.join(change_my_view_directory, 'all_data.pkl'))

# units.to_csv(os.path.join(change_my_view_directory, 'units.csv'))
# units.to_pickle(os.path.join(change_my_view_directory, 'units.pkl'))

delta = filter_results.loc[filter_results['delta'] == 1]
print('delta size:',  delta.shape)
# after_17 = filter_results.loc[filter_results['comment_created_utc'] >= 1475280000]
# print(after_17.shape)
# delta_after_17 = after_17.loc[filter_results['delta'] == 1]
# print(delta_after_17.shape)
# no_delta_after_17 = after_17.loc[filter_results['delta'] == 0]
# print(no_delta_after_17.shape)

sub_delta = delta['submission_id']
sub_delta = sub_delta.drop_duplicates()
sub_delta = list(sub_delta)
# after_17 = filter_results.loc[(filter_results['comment_created_utc'] >= 1483228800) & (filter_results['delta'] == 0)]
no_delta_from_sub_delta = filter_results.loc[(filter_results['submission_id'].isin(sub_delta)) &
                                             (filter_results['delta'] == 0)]
final_results = pd.concat([delta, no_delta_from_sub_delta])

print('no_delta_from_sub_delta size:', no_delta_from_sub_delta.shape)
print('final result size:', final_results.shape)

final_results.to_csv(os.path.join(change_my_view_directory, 'all_data_0304.csv'))

units = final_results.loc[(final_results['comment_author'] != final_results['submission_author'])]
print(units.shape)

units.to_csv(os.path.join(change_my_view_directory, 'units_0304.csv'))
units.to_pickle(os.path.join(change_my_view_directory, 'units_0304.pkl'))
#
# group_comment_author = units.groupby('comment_author')['comment_id']
# group_comment_author_count = group_comment_author.count()
# group_comment_author_count = pd.DataFrame(group_comment_author_count)
# group_comment_author_count = group_comment_author_count.assign(comment_author=group_comment_author_count.index)
# group_comment_author_count.columns = ['comments_count', 'comment_author']
# final_results_with_count = final_results.merge(group_comment_author_count, on='comment_author')
#
# final_results_with_count_filter = final_results_with_count.loc[final_results_with_count['comments_count'] > 1]