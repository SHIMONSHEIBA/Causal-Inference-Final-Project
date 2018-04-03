import pandas as pd
import os


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")

final_results = pd.read_csv(os.path.join(data_directory, 'filter_comments_submissions_03041135.csv'))

group_comment_author = final_results.groupby('comment_author')['comment_id']
group_comment_author_count = group_comment_author.count()
group_comment_author_count = pd.DataFrame(group_comment_author_count)
group_comment_author_count = group_comment_author_count.assign(comment_author=group_comment_author_count.index)
group_comment_author_count.columns = ['comments_count', 'comment_author']
final_results_with_count = final_results.merge(group_comment_author_count, on='comment_author')

final_results_with_count_filter = final_results_with_count.loc[final_results_with_count['comments_count'] > 1]

reut = 1