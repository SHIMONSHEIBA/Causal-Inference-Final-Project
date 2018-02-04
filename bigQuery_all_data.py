import os
import logging
from pandas.io import gbq
import pandas as pd
import time
from datetime import datetime

base_directory = os.path.abspath(os.curdir)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(base_directory, 'reut shimon project-da938df278f6.json')
MyProjectID = 'reut-shimon-project'

seconds_in_week = 604800
seconds_in_month = 2592000


def second_query(predicted_results, number_of_not_passed_threshold):
    is_post_before_df_total = pd.DataFrame()
    for index, comment in predicted_results.iterrows():
        if comment['classifier_result'] > 0.9:
            # query: check if the author of the submission posted in the recommended subreddit before the reference
            is_post_before_query = """SELECT COUNT(*) AS NUM_OF_POSTS, MIN(created_utc) AS MIN_DATE,
                                    MAX(created_utc) AS MAX_DATE
                                    FROM
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2015_12] 
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_01]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_02]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_03]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_04]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_05]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_06]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_07]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_08]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_09]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_10]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_11]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_12]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_01]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_02]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_01]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_02]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_03]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_04]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_05]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_06]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_07]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_08]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_09]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_10]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_11]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_12]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_01]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_02]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_03]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_04]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_05]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_06]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_07]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_08]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_09]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_10]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_11]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_12]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_01]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_02]
                                    WHERE author = '{0}' AND created_utc < {1} AND subreddit = '{2}')"""\
                .format(comment['submission_author'], comment['comment_created_time'], comment['recommend_subreddit'])

            print('{}: Start quering the query: {} '.
                  format((time.asctime(time.localtime(time.time()))), is_post_before_query))
            logging.debug('{}: Start quering the query: {}'.
                          format((time.asctime(time.localtime(time.time()))), is_post_before_query))
            # run the query
            is_post_before_df = (gbq.read_gbq(is_post_before_query, project_id=MyProjectID))
            # add relevant columns
            is_post_before_df = is_post_before_df.assign(sub_com_index=comment['index'])
            is_post_before_df = is_post_before_df.assign(referral_utc=comment['comment_created_time'])
            is_post_before_df = is_post_before_df.assign(classifier_result=comment['classifier_result'])
            is_post_before_df = is_post_before_df.assign(time_since_last_post=
                                                         is_post_before_df['MAX_DATE'] -
                                                         is_post_before_df['comment_created_time'])

            if is_post_before_df.empty:
                is_post_before_df = pd.DataFrame({'NUM_OF_POSTS': 0, 'MIN_DATE': 0, 'MAX_DATE': 0,
                                                  'sub_com_index': comment['index'],
                                                  'referral_utc': comment['comment_created_time'],
                                                  'classifier_result': comment['classifier_result'],
                                                  'time_since_last_post': 0}, index=[1])

            is_post_before_df_total = pd.concat([is_post_before_df_total, is_post_before_df], axis=0)
            is_post_before_df_total.to_csv("is_post_before_df_total.csv", encoding='utf-8')

            # TODO: change is_written_in_last_week to month and comment['count'] to the correct
            # only users that didn't write before, or write less than 3 posts and didn't write in the last week will be
            # checked:
            if is_post_before_df['NUM_OF_POSTS'] < 4 and is_post_before_df['time_since_last_post'] > seconds_in_week:
                # if is_post_before_df.empty:
                # query: check if the author of the submission posted in the recommended subreddit after the reference
                is_post_after_query = """SELECT COUNT(*) AS NUM_OF_POSTS FROM
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2015_12] 
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_01]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_02]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_03]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_04]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_05]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_06]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_07]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_08]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_09]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_10]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_11]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_12]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_01]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_02]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_01]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_02]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_03]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_04]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_05]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_06]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_07]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_08]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_09]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_10]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_11]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_12]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_01]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_02]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_03]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_04]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_05]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_06]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_07]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_08]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_09]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_10]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_11]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_12]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_01]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}'),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_02]
                                    WHERE author = '{0}' AND created_utc > {1} AND subreddit = '{2}')""".format(
                    comment['submission_author'], comment['comment_created_time'], comment['recommend_subreddit'])

                print('{}: Start quering the query: {} '.
                      format((time.asctime(time.localtime(time.time()))), is_post_after_query))
                logging.debug('{}: Start quering the query: {}'.
                              format((time.asctime(time.localtime(time.time()))), is_post_after_query))
                # run the query
                is_post_after_df = (gbq.read_gbq(is_post_after_query, project_id=MyProjectID))

                if is_post_after_df.empty:
                    # the author of the submission didn't post in the recommended subreddit before,
                    # or posted few times according to our threshold, and didn't post after the ref
                    # --> case 1 and case 3
                    comment['IsEfficient'] = -1
                    predicted_results = predicted_results.append(comment)
                else:  # the author of the submission posted in the recommended subreddit after the ref and not before
                    # or posted few times before- according to our threshold
                    # --> case 2 and case 4
                    comment['IsEfficient'] = 1
                    predicted_results = predicted_results.append(comment)

            else:  # didn't pass the threshold
                number_of_not_passed_threshold += 1
                print('eliminate comment index: {}, count is:{}, time between comment and posts is:{}'
                      'and last post before is: {}'.
                      format(comment['index'], is_post_before_df['NUM_OF_POSTS'],
                             is_post_before_df['time_since_last_post'], is_post_before_df['MAX_DATE']))
            predicted_results.to_csv("predictedResultsEfficiency.csv", encoding='utf-8')
    predicted_results.to_csv("predictedResultsEfficiency.csv", encoding='utf-8')
    print('number_of_not_passed_threshold is: {}'.format(number_of_not_passed_threshold))
    return predicted_results


if __name__ == '__main__':
    number_of_not_passed_threshold = 0
    logging.basicConfig(filename=datetime.now().strftime('LogFile_%d_%m_%Y_%H_%M.log'), level=logging.DEBUG)
    print('{}: Start quering'.format((time.asctime(time.localtime(time.time())))))
    logging.info('{}: Start quering'.format((time.asctime(time.localtime(time.time())))))

    resultsPredicted = pd.read_excel('result_to_classify_small.xlsx')
    resultsPredictedSort = resultsPredicted.sort_values(by=['classifier_result'], ascending=False, axis=0)
    FinalResultsWithEfficient = second_query(resultsPredictedSort, number_of_not_passed_threshold)
