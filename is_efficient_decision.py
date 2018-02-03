import pandas as pd

seconds_in_week = 604800
seconds_in_month = 2592000


def is_written_in_last_week(row):
    if row['time_since_last_post'] < seconds_in_week:
        return True
    else:
        return False


def is_written_in_last_month(row):
    if row['time_since_last_post'] < seconds_in_month:
        return True
    else:
        return False


all_data = pd.read_csv('is_post_before_df_total_fixed_final.csv')
resultsPredicted = pd.read_excel('FinalFeatures_with_comment_time.xlsx')

index_result = all_data['sub_comment_index']

# only those who wrote before
all_data_write = all_data[~all_data.created_utc.isin(['not write'])]
# only those who didn't write before
all_data_not_write = all_data[all_data.created_utc.isin(['not write'])]
count_not_write = all_data_not_write[['sub_comment_index', 'referral_utc']]
count_not_write = count_not_write.assign(count=0)
count_not_write = count_not_write.assign(time_since_last_post=0)
count_not_write = count_not_write.assign(is_written_in_last_week=False)
count_not_write = count_not_write.assign(is_written_in_last_month=False)
count_not_write = count_not_write.assign(max_created_utc=count_not_write['referral_utc'])
# the number of times the user wrote before - if he wrote
count_before = all_data_write.groupby(['sub_comment_index']).size()
count_before = pd.DataFrame(count_before)
count_before['sub_comment_index'] = count_before.index
count_before.columns = ['count', 'sub_comment_index']

# the number of times the user wrote before - if he wrote
all_data_write.loc[:, 'created_utc'] = all_data_write.loc[:, 'created_utc'].astype(int)
max_before = all_data_write.groupby(['sub_comment_index'])['created_utc'].max()
max_before = pd.DataFrame(max_before)

data_to_join = all_data_write[['sub_comment_index', 'referral_utc']].drop_duplicates()

# create DF with sub_comment_index, referral_utc and max created_utc
max_before_with_time = data_to_join.join(max_before, on='sub_comment_index')
max_before_with_time.columns = ['sub_comment_index', 'referral_utc', 'max_created_utc']

# create column is_written_in_last_week
max_before_with_time = max_before_with_time.assign(time_since_last_post=lambda x: x.referral_utc - x.max_created_utc)
max_before_with_time['is_written_in_last_week'] = max_before_with_time.apply(lambda x: is_written_in_last_week(x),
                                                                             axis=1)
max_before_with_time['is_written_in_last_month'] = max_before_with_time.apply(lambda x: is_written_in_last_month(x),
                                                                              axis=1)

# join count and max_before_with_time
count_before = count_before[['count']]
write_before = max_before_with_time.join(count_before, on='sub_comment_index')

# all the comments with is_efficient -1 the number ot posts written before in the recommended subreddit, the seconds
# between the referral and the last comments and is_written_in_last_week
info_per_ref = pd.concat([write_before, count_not_write], axis=0)
info_per_ref.to_csv('info_per_ref.csv')

# statistics:
print('total number of ref been checked is: {}'.format(len(info_per_ref.index)))
print('number of users not written before is: {} --> stay'.format(len(count_not_write.index)))
checked_list_last_week = [1, 2, 3, 4, 5, 6]
checked_list_more_than = [0, 1, 2, 3, 4, 5, 6]
for eliminate in [6, 5, 4]:
    checked_list_last_week.remove(eliminate)
    checked_list_more_than.remove(eliminate)
    freq_number_of_posts = info_per_ref.groupby(['count']).size()
    less_than_three_posts = info_per_ref[info_per_ref['count'].isin(checked_list_last_week)]
    freq_wrote_last_week = less_than_three_posts.groupby(['is_written_in_last_week']).size()
    freq_wrote_last_month = less_than_three_posts.groupby(['is_written_in_last_month']).size()
    print('number of ref with less than {} posts in the recommended subreddit is: {} --> stay'.
          format(eliminate-1, len(less_than_three_posts.index)))
    more_than_three_posts = info_per_ref[~info_per_ref['count'].isin(checked_list_more_than)]
    print('number of ref with more than {} posts in the recommended subreddit is: {} --> these will be eliminated'.
          format(eliminate, len(more_than_three_posts.index)))
    wrote_in_last_week = less_than_three_posts[less_than_three_posts['is_written_in_last_week'] == True]
    wrote_in_last_month = less_than_three_posts[less_than_three_posts['is_written_in_last_month'] == True]
    print('number of ref with less than {} posts in the recommended subreddit that wrote in the last week is: {} --> '
          'these will be eliminated'.format(eliminate-1, len(wrote_in_last_week.index)))
    print('number of ref with less than {} posts in the recommended subreddit that wrote in the last month is: {} --> '
          'these will be eliminated'.format(eliminate-1, len(wrote_in_last_month.index)))
