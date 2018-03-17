import pandas as pd
import numpy as np
from time import time
import time
import itertools
import math
from collections import Counter
from time import strftime, localtime
from datetime import datetime
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from optparse import OptionParser
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


op = OptionParser()
op.add_option('--efficient_threshold',
              action='store', type=int, default=0.8,
              help='threshold for efficient reference according to classifier score')
op.add_option('--use_date_threshold',
              action='store', default=True,
              help='take only data about comments and submissions that published before the comment')
op.add_option('--pickel_not_saved',
              action='store', default=False,
              help='did I have already saved the subreddit_dict into pickel --> just need to load it')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")


class CreateFeatures:
    """
    This class will build the features for each unit: comment and its submission
    """
    def __init__(self):
        # Load all relevant data
        self.units = pd.read_csv(os.path.join(data_directory, 'data_label_treatment.csv'))
        pd.to_numeric(self.units['submission_created_time'])
        pd.to_numeric(self.units['comment_created_time'])
        self.all_data = pd.read_csv(os.path.join(data_directory, 'all_comments_submissions.csv'))
        pd.to_numeric(self.all_data['comment_created_time'])
        pd.to_numeric(self.all_data['submission_created_time'])

    def number_of_message(self, user, comment_time, messages_type):
        """
        Get the number of messages (submissions and comments) the user posted we have in the data
        :param str user: the user name we want to check
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param str messages_type: submission / comment - what we want to check
        :return: the number of messages of the messages_type
        :rtype int
        """

        data = self.all_data[[messages_type + '_author', messages_type + '_id', messages_type + '_created_utc']]
        data = data.drop_duplicates()
        relevant_data = data.loc[(data[messages_type + '_author'] == user)
                                 & (data[messages_type + '_created_utc'] < comment_time)]
        number_of_posts = relevant_data.shape[0]

        return number_of_posts

    def comment_in_tree(self, user, comment_time, submission_id):
        """
        Check if this is the first comment the comment author posted for this submission
        :param str user: the user name we want to check
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param int submission_id: the submission id
        :return: int is_first_comment_in_tree: 1 - if this is the first time, 0 - otherwise
                int number_of_comments_in_tree: number of comments he wrote in the submission tree until time t
        """

        all_comments_user_in_tree = self.all_data.loc[(self.all_data['comment_author'] == user)
                                                      & (self.all_data['comment_created_utc'] < comment_time)
                                                      & (self.all_data['submission_id'] == submission_id)]
        if all_comments_user_in_tree.empty:
            number_of_comments_in_tree = 0
            is_first_comment_in_tree = 1
        else:
            number_of_comments_in_tree = all_comments_user_in_tree.shape[0]
            is_first_comment_in_tree = 0

        return is_first_comment_in_tree, number_of_comments_in_tree

    def submitter_respond_to_comment_user(self, comment_user, submitter, comment_time, submission_id):
        """
        Calculate the number of comments the submitter posted as a response to the comment author in this submission
        :param str comment_user: the comment user
        :param str submitter: the submission user
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param int submission_id: the submission id
        :return: int number_of_respond_by_submitter: the number of responds by the submitter to the comment user
                int number_of_respond_by_submitter_total: the number of responds by the submitter in total
        """

        all_comments_submitter_in_tree = self.all_data.loc[(self.all_data['comment_author'] == submitter)
                                                           & (self.all_data['comment_created_utc'] < comment_time)
                                                           & (self.all_data['submission_id'] == submission_id)]
        # the parent ids of all the comments that were written by the submitter
        parent_id_list = list(all_comments_submitter_in_tree['parent_id'])
        # take all comments in this submission that were written by the comment user and are the parents of the
        # submitter's comments - i.e the submitter respond to the comment user
        parent_by_the_comment_author = self.all_data.loc[(self.all_data['comment_id'].isin(parent_id_list))
                                                         & (self.all_data['comment_author'] == comment_user)
                                                         & (self.all_data['comment_created_utc'] < comment_time)
                                                         & (self.all_data['submission_id'] == submission_id)]

        # number of responses by submitter : parent_id != submission_id
        respond_by_submitter_total =\
            all_comments_submitter_in_tree.loc[all_comments_submitter_in_tree['parent_id'] != submission_id]

        number_of_respond_by_submitter = parent_by_the_comment_author.shape[0]
        number_of_respond_by_submitter_total = respond_by_submitter_total.shape[0]

        return number_of_respond_by_submitter, number_of_respond_by_submitter_total

    def time_to_first_comment(self, submission_id, submission_created_time):
        """
        Calculate the time between the submission and the first comment
        :param int submission_id: the submission id
        :return: int the seconds between the submission and the first comment in its tree
        """

        all_submission_comments = self.all_data.loc[self.all_data['submission_id'] == submission_id]
        time_of_first_comment = all_submission_comments['comment_created_utc'].min()
        time_until_first_comment = time_of_first_comment - submission_created_time

        return time_until_first_comment


def main():
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    create_features = CreateFeatures()
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: all data: {}, units data: {}'.format(create_features.all_data.shape,
                                                            create_features.units.shape))

    # Features calculated for all the data frame:
    create_features.units['submission_len'] = create_features.units['submission_body'].str.len()
    create_features.units['title_len'] = create_features.units['submission_title'].str.len()

    all_comments_features = pd.DataFrame()
    for index, comment in create_features.units.iterrows():
        if index % 100 == 0:
            print('{}: Finish calculate {} samples'.format((time.asctime(time.localtime(time.time()))), index))
        comment_author = comment['comment_author']
        comment_time = comment['comment_created_utc']
        submission_time = comment['submission_created_utc']
        submission_id = comment['submission_id']
        submission_num_comments = comment['submission_num_comments']

        # Get comment author features:
        comment_author_number_submission = create_features.number_of_message(comment_author, comment_time, 'submission')
        comment_author_number_comment = create_features.number_of_message(comment_author, comment_time, 'comment')

        # Get submission author features:
        submission_author = comment['submission_author']
        sub_author_number_submission = create_features.number_of_message(submission_author, comment_time, 'submission')
        sub_author_number_comment = create_features.number_of_message(submission_author, comment_time, 'comment')

        # Get the time between the submission and the comment time and the ration between the first comment:
        time_to_comment = comment['time_between']
        time_between_messages_hour = math.floor(time_to_comment/3600.0)
        time_between_messages_min = math.floor((time_to_comment - 3600*time_between_messages_hour)/60.0)/100.0
        time_between_messages = time_between_messages_hour + time_between_messages_min
        time_until_first_comment = create_features.time_to_first_comment(submission_id, submission_time)
        time_ratio = comment_time - time_until_first_comment

        # Comment features:
        is_first_comment_in_tree, number_of_comments_in_tree_by_comment_user =\
            create_features.comment_in_tree(comment_author, comment_time, submission_id)
        _, number_of_comments_in_tree_from_submitter =\
            create_features.comment_in_tree(submission_author, comment_time, submission_id)
        number_of_respond_by_submitter, number_of_respond_by_submitter_total =\
            create_features.submitter_respond_to_comment_user(comment_author, submission_author,
                                                              comment_time, submission_id)

        # Ratio of comments number:
        respond_to_comment_user_all_ratio = number_of_respond_by_submitter / submission_num_comments
        respond_to_comment_user_responses_ratio =\
            number_of_respond_by_submitter / number_of_respond_by_submitter_total
        respond_total_ratio = number_of_respond_by_submitter_total / submission_num_comments

        features = [comment_author_number_submission, comment_author_number_comment, sub_author_number_submission,
                    sub_author_number_comment, time_between_messages, time_ratio, is_first_comment_in_tree,
                    number_of_comments_in_tree_by_comment_user, number_of_comments_in_tree_from_submitter,
                    number_of_respond_by_submitter, number_of_respond_by_submitter_total,
                    respond_to_comment_user_all_ratio, respond_to_comment_user_responses_ratio, respond_total_ratio]

        labels = ('comment_author_number_submission', 'comment_author_number_comment', 'sub_author_number_submission',
                  'sub_author_number_comment', 'time_between_messages', 'time_ratio', 'is_first_comment_in_tree',
                  'number_of_comments_in_tree_by_comment_user', 'number_of_comments_in_tree_from_submitter',
                  'number_of_respond_by_submitter', 'number_of_respond_by_submitter_total',
                  'respond_to_comment_user_all_ratio', 'respond_to_comment_user_responses_ratio', 'respond_total_ratio')

        featuresDF = pd.Series(features, index=labels)

        comment_features = comment.append(featuresDF)
        all_comments_features = pd.concat([all_comments_features, comment_features], axis=1)
        all_comments_features.T.to_csv(os.path.join(data_directory, 'features_CMV.csv'), encoding='utf-8')

    # export the data to csv file
    all_comments_features.T.to_csv(os.path.join(data_directory, 'features_CMV.csv'), encoding='utf-8')


if __name__ == '__main__':
    main()

