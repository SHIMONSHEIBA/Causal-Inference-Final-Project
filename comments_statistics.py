import pandas as pd
import os

base_directory = os.path.abspath(os.curdir)
change_my_view_directory = os.path.join(base_directory, 'change my view')
comments = pd.read_csv(os.path.join(change_my_view_directory, 'all submissions comments with label.csv'))
submissions = pd.read_excel(os.path.join(change_my_view_directory, 'all submissions.xlsx'))

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

group_depth_count = group_depth.count()
group_len_count = group_len.count()
group_time_count = group_time.count()

filter_results = join_result.loc[(join_result['time_between'] < 10080.0) & (join_result['comment_len'] > 150)
                                 & (join_result['comment_author'] != 'DeltaBot')
                                 & (~join_result['comment_body'].str.contains('your comment has been removed:'))
                                 & (join_result['comment_body'].str.contains('[deleted]'))]
print(filter_results.shape)
filter_results.to_csv(os.path.join(change_my_view_directory, 'filter_comments_submissions.csv'))
