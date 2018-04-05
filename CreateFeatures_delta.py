import pandas as pd
from time import time
import time
import math
from datetime import datetime
import logging
import pytz
from copy import copy
import os
import nltk as nk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.sklearn_api import ldamodel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")

log_directory = os.path.join(base_directory, 'logs')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_create_features_delta_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )

# for topic modeling clean text
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


class CreateFeatures:
    """
    This class will build the features for each unit: comment and its submission
    """
    def __init__(self):
        # Load all relevant data
        units_columns = ['comment_body', 'comment_author', 'submission_author', 'submission_body', 'submission_title',
                         'comment_id', 'parent_id', 'comment_created_utc', 'submission_created_utc', 'submission_id',
                         'submission_num_comments', 'time_between']

        # all the units, with label
        self.units = pd.read_csv(os.path.join(data_directory, 'units_0304.csv'), skipinitialspace=True,
                                 usecols=units_columns)
        # self.units = pd.read_excel(os.path.join(data_directory, 'small_data.xlsx'), skipinitialspace=True,
        #                          usecols=units_columns)
        pd.to_numeric(self.units['submission_created_utc'])
        pd.to_numeric(self.units['comment_created_utc'])
        self.units['parent_id'] = self.units.parent_id.str.replace("b't1_", '', count=1)
        self.units['parent_id'] = self.units.parent_id.str.replace("b't3_", '', count=1)
        self.units['comment_id'] = self.units.comment_id.str.replace("b'", '', count=1)
        self.units['parent_id'] = self.units.parent_id.str.replace("'", '', count=1)
        self.units['comment_id'] = self.units.comment_id.str.replace("'", '', count=1)

        self.units.assign(commenter_number_submission='', commenter_number_comment='',
                          submitter_number_submission='', submitter_number_comment='', time_ratio='',
                          time_until_first_comment='', time_between_comment_first_comment='',
                          is_first_comment_in_tree='', number_of_comments_in_tree_by_comment_user='',
                          number_of_comments_in_tree_from_submitter='', number_of_respond_by_submitter='',
                          number_of_respond_by_submitter_total='', respond_to_comment_user_all_ratio='',
                          respond_to_comment_user_responses_ratio='', respond_total_ratio='',
                          submitter_seniority_days='', commenter_seniority_days='', time_between_messages='')

        all_data_columns = ['comment_body', 'comment_author', 'submission_author', 'submission_body', 'submission_title',
                            'comment_id', 'parent_id', 'comment_created_utc', 'submission_created_utc', 'submission_id',
                            'submission_num_comments', 'time_between']
        self.all_data = pd.read_csv(os.path.join(data_directory, 'all_data_0304.csv'), skipinitialspace=True,
                                    usecols=all_data_columns)
        # self.all_data = pd.read_excel(os.path.join(data_directory, 'small_data.xlsx'), skipinitialspace=True,
        #                          usecols=units_columns)
        pd.to_numeric(self.all_data['comment_created_utc'])
        pd.to_numeric(self.all_data['submission_created_utc'])
        self.all_data['parent_id'] = self.all_data.parent_id.str.replace("b't1_", '', count=1)
        self.all_data['parent_id'] = self.all_data.parent_id.str.replace("b't3_", '', count=1)
        self.all_data['comment_id'] = self.all_data.comment_id.str.replace("b'", '', count=1)
        self.all_data['parent_id'] = self.all_data.parent_id.str.replace("'", '', count=1)
        self.all_data['comment_id'] = self.all_data.comment_id.str.replace("'", '', count=1)

        # data after drop duplications:
        self.submission_no_dup = self.all_data[['submission_author', 'submission_id', 'submission_created_utc']]
        self.submission_no_dup = self.submission_no_dup.drop_duplicates()

        # create dict with the data for each submission:
        start_time = time.time()
        self.submission_data_dict = dict()
        submission_ids = self.submission_no_dup['submission_id']
        for index, submission_id in submission_ids.iteritems():
            self.submission_data_dict[submission_id] = self.all_data.loc[self.all_data['submission_id'] == submission_id]
        print('time to create submission dict: ', time.time() - start_time)

    def number_of_message(self, user, comment_time, messages_type):
        """
        Get the number of messages (submissions and comments) the user posted we have in the data
        :param str user: the user name we want to check
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param str messages_type: submission / comment - what we want to check
        :return: the number of messages of the messages_type
        :rtype int
        """
        # data = self.all_data[[messages_type + '_author', messages_type + '_id', messages_type + '_created_utc']]
        # data = data.drop_duplicates()
        # after_drop_time = time.time()
        if messages_type == 'comment':
            relevant_data = self.all_data.loc[(self.all_data[messages_type + '_author'] == user)
                                              & (self.all_data[messages_type + '_created_utc'] < comment_time)]
        else:
            relevant_data =\
                self.submission_no_dup.loc[(self.submission_no_dup[messages_type + '_author'] == user)
                                           & (self.submission_no_dup[messages_type + '_created_utc'] < comment_time)]
        number_of_posts = relevant_data.shape[0]

        return number_of_posts

    def comment_in_tree(self, user, comment_time, submission_id, comment_user=None,
                        submitter_respond_to_comment_user=False):
        """
        Check if this is the first comment the comment author posted for this submission
        :param str user: the user name we want to check (either submitter or comment user)
        :param int comment_time: the time the comment in the unit was posted (time t)
        :param int submission_id: the submission id
        :param str comment_user: the comment user name of this unit
        :param bool submitter_respond_to_comment_user: whether we check the submitter_respond_to_comment_user or not
        :return: int is_first_comment_in_tree: 1 - if this is the first time, 0 - otherwise
                int number_of_comments_in_tree: number of comments he wrote in the submission tree until time t
                int number_of_respond_by_submitter: the number of responds by the submitter to the comment user
                int number_of_respond_by_submitter_total: the number of responds by the submitter in total
        """

        submission_data = self.submission_data_dict[submission_id]
        all_comments_user_in_tree = submission_data.loc[(submission_data['comment_author'] == user)
                                                        & (submission_data['comment_created_utc'] < comment_time)]
        if all_comments_user_in_tree.empty:
            number_of_comments_in_tree = 0
            is_first_comment_in_tree = 1
            # if there are no comments before comment_time - if this is the submitter, no need to check the
            # number_of_respond_by_submitter and number_of_respond_by_submitter_total - they will be 0
            number_of_respond_by_submitter = 0
            number_of_respond_by_submitter_total = 0
            return is_first_comment_in_tree, number_of_comments_in_tree, number_of_respond_by_submitter,\
                   number_of_respond_by_submitter_total
        else:  # if there are comments in before comment_time from this user
            number_of_comments_in_tree = all_comments_user_in_tree.shape[0]
            is_first_comment_in_tree = 0

            if not submitter_respond_to_comment_user:  # if this the comment user
                return is_first_comment_in_tree, number_of_comments_in_tree, 0, 0
            else:  # submitter = user
                # the parent ids of all the comments that were written by the submitter in this submission until time t
                parent_id_list = list(all_comments_user_in_tree['parent_id'])
                # take all comments in this submission that were written by the comment user and are the parents of the
                # submitter's comments - i.e the submitter respond to the comment user
                parent_by_the_comment_author = submission_data[(submission_data['comment_id'].isin(parent_id_list))
                                                               & (submission_data['comment_author'] == comment_user)
                                                               & (submission_data['comment_created_utc'] < comment_time)]

                # number of responses by submitter : parent_id != submission_id -
                # the submitter wrote a comment to someone - respond
                respond_by_submitter_total =\
                    all_comments_user_in_tree.loc[all_comments_user_in_tree['parent_id'] != submission_id]

                number_of_respond_by_submitter = parent_by_the_comment_author.shape[0]
                number_of_respond_by_submitter_total = respond_by_submitter_total.shape[0]

                return is_first_comment_in_tree, number_of_comments_in_tree, number_of_respond_by_submitter,\
                        number_of_respond_by_submitter_total

    def time_to_first_comment(self, submission_id, submission_created_time, comment_created_time):
        """
        Calculate the time between the submission and the first comment
        :param int submission_id: the submission id
        :param int submission_created_time: the utc time of the submission
        :param int comment_created_time: the utc time of the comment
        :return: int the seconds between the submission and the first comment in its tree
        """

        all_submission_comments = self.submission_data_dict[submission_id]
        time_of_first_comment = all_submission_comments['comment_created_utc'].min()
        time_until_first_comment = time_of_first_comment - submission_created_time
        time_between_comment_first_comment = comment_created_time - time_of_first_comment

        return time_until_first_comment, time_between_comment_first_comment

    def calculate_user_seniority(self, user):
        """
        Calculate the user seniority in change my view subreddit in days
        :param str user: the user name
        :return: int the number of days since the first post of the user (submissions and comments)
        """
        user_all_comments = self.all_data.loc[self.all_data['comment_author'] == user]
        user_all_submissions = self.all_data.loc[self.all_data['submission_author'] == user]
        first_comment_time = user_all_comments.comment_created_utc.min()
        first_submission_time = user_all_submissions.submission_created_utc.min()
        if not user_all_comments.empty:  # the user has not write any comment - so the min will be nan
            first_post_time = min(first_comment_time, first_submission_time)
        else:
            if not user_all_submissions.empty:
                first_post_time = min(first_submission_time, first_comment_time)
            else:
                print('no comments and submission for user: {}'.format(user))
                logging.info('no comments and submission for user: {}'.format(user))
                return 0
        tz = pytz.timezone('GMT')  # America/New_York
        date_comment = datetime.fromtimestamp(first_post_time, tz)
        utc_now = int(time.time())
        date_now = datetime.fromtimestamp(utc_now, tz)
        time_between = date_now - date_comment

        return time_between.days

    def loop_over_comment_for_quote(self, comment, comment_body):
        """
        Go over the comment and check if there is quote in each part
        :param pandas series comment: a series with all the comment's information
        :param str comment_body: the comment's body
        :return: 0 if there is no quote in this part of comment, 1 if there is, -1 if we don't want this unit
        """
        is_quote = 0
        if '>' in comment_body:  # this is the sign for a quote (|) in the comment
            while '>' in comment_body and not is_quote:
                quote_index = comment_body.find('>')
                comment_body = comment_body[quote_index + 1:]
                is_quote = self.check_quote(comment, comment_body)
        else:  # there is not quote at all
            is_quote = 0

        return is_quote

    def check_quote(self, comment, comment_body):
        """
        Check if there is a quote in the comment.
        Check if it is a quote of a comment of the submitter or the submission itself
        :param pandas series comment: a series with all the comment's information
        :param str comment_body: the comment's body
        :return: 0 if there is no quote in this part of comment, 1 if there is, -1 if we don't want this unit
        """

        no_parent = False
        quote = copy(comment_body)
        nn_index = quote.find('\\n')
        n_index = quote.find('\n')
        if nn_index == -1:  # there is no \\n
            quote = quote
        elif n_index != -1:  # there is \n
            quote = quote[: n_index - 1]
        else:
            quote = quote[: nn_index - 1]  # take the quote: after the > and until the first \n
        # parse the parent id
        parent_id = comment['parent_id']

        # if the parent is the submission - take the submission body
        if parent_id == comment['submission_id']:
            parent_body = comment['submission_body']
            parent_author = comment['submission_author']
        else:  # if not - get the parent
            # parent_id = "b'" + parent_id + "'"
            parent = self.all_data.loc[self.all_data['comment_id'] == parent_id]
            if parent.empty:  # if we don't have the parent as comment in the data
                # print('no parent comment for comment_id: {}'.format(comment_id))
                # logging.info('no parent comment for comment_id: {}'.format(comment_id))
                parent_body = ''
                parent_author = ''
                no_parent = True
            else:
                parent = pd.Series(parent.iloc[0])
                parent_body = parent['comment_body']
                parent_author = parent['comment_author']
        submission_author = comment['submission_author']
        submission_body = comment['submission_body']
        submission_title = comment['submission_title']

        if submission_author == parent_author:  # check if the parent author is the submitter
            # if he quote the submission or the parent
            if (quote in parent_body) or (quote in submission_body) or (quote in submission_title):
                # self.units.loc[index, 'treated'] = 1
                return 1
            else:  # he didn't quote the submitter
                # self.units.loc[index, 'treated'] = 0
                return 0
        else:  # if the parent author is not the submitter
            if (quote in submission_body) or (quote in submission_title):  # we only care of he quote the submission:
                # self.units.loc[index, 'treated'] = 1
                # print('quote the submission, but it is not its parent for comment_id: {}'.format(comment_id))
                # logging.info('quote the submission, but it is not its parent for comment_id: {}'.format(comment_id))
                return 1
            else:
                if no_parent:
                    # if there is no parent and he didn't quote the submission, we can't know if he quote the parent
                    # - so maybe we don't need to use it
                    # self.units.loc[index, 'treated'] = -1
                    return -1
                else:
                    # self.units.loc[index, 'treated'] = 0
                    return 0


def sentiment_analysis(text):
    """
    This function calculate the sentiment of a text. It is calculate the probability of the text to be negative,
    positive or neutral
    :param str text: the text we calculate its sentiments
    :return: list: pos_prob, neg_prob, neutral_prob
    """
    sid = SentimentIntensityAnalyzer()
    result = sid.polarity_scores(text)
    neg_prob = result['neg']
    neutral_prob = result['neu']
    pos_prob = result['pos']
    return [pos_prob, neg_prob, neutral_prob]


def get_POS(text):
    """
    This function find the POS of each word in text.
    :param str text: the text we want to find its POS
    :return: list(tuple) - a list of the words- for each a tuple: (word, POS)
    """
    text_parsed = nk.word_tokenize(text)
    words_pos = nk.pos_tag(text_parsed)

    return words_pos


def percent_of_adj(text):
    """
    This function calculate the % of the adjectives in the text
    :param str text: the text we want to calculate its %
    :return: float: number_adj_pos/number_all_pos in the text
    """
    pos_text = get_POS(text)
    pos_df = pd.DataFrame(pos_text, columns=['word', 'POS'])
    number_all_pos = pos_df.shape[0]
    all_pos = pos_df['POS']
    freq = nk.FreqDist(all_pos)
    number_adj_pos = freq['JJ'] + freq['JJS'] + freq['JJR']
    percent_of_adj_pos = number_adj_pos/number_all_pos

    return percent_of_adj_pos


def clean(text):
    """
    This function clean a text from stop words and punctuations and them lemmatize the words
    :param str text: the text we want to clean
    :return: str normalized: the cleaned text
    """
    text = text.lstrip('b').strip('"').strip("'").strip(">")
    stop_free = " ".join([i for i in text.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def main():
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    create_features = CreateFeatures()
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: all data: {}, units data: {}'.format(create_features.all_data.shape,
                                                            create_features.units.shape))

    # Features calculated for all the data frame:
    create_features.units['comment_len'] = create_features.units['comment_body'].str.len()
    create_features.units['submission_len'] = create_features.units['submission_body'].str.len()
    create_features.units['title_len'] = create_features.units['submission_title'].str.len()

    all_comments_features = pd.DataFrame()
    new_index = 0
    number_of_treatment_minus_1 = 0
    for index, comment in create_features.units.iterrows():
        if new_index % 100 == 0:
            print('{}: Finish calculate {} samples'.format((time.asctime(time.localtime(time.time()))), new_index))
        comment_author = copy(comment['comment_author'])
        comment_time = copy(comment['comment_created_utc'])
        submission_time = copy(comment['submission_created_utc'])
        submission_id = copy(comment['submission_id'])
        submission_num_comments = copy(comment['submission_num_comments'])
        comment_body = copy(comment['comment_body'])
        submission_body = copy(comment['submission_body'])
        title = copy(comment['submission_title'])

        # treatment:
        is_quote = create_features.loop_over_comment_for_quote(comment, comment_body)
        if is_quote != -1:
            create_features.units.loc[index, 'treated'] = is_quote
        else:
            print('{}: treatment = -1'.format((time.asctime(time.localtime(time.time())))))
            number_of_treatment_minus_1 += 1
            continue

        # Get comment author features:
        # print('{}: Get comment author features'.format((time.asctime(time.localtime(time.time())))))
        create_features.units.loc[index, 'commenter_number_submission'] =\
            create_features.number_of_message(comment_author, comment_time, 'submission')
        create_features.units.loc[index, 'commenter_number_comment'] =\
            create_features.number_of_message(comment_author, comment_time, 'comment')
        create_features.units.loc[index, 'commenter_seniority_days'] =\
            create_features.calculate_user_seniority(comment_author)

        # Get submission author features:
        # print('{}: Get submission author features'.format((time.asctime(time.localtime(time.time())))))
        submission_author = comment['submission_author']
        create_features.units.loc[index, 'submitter_number_submission']\
            = create_features.number_of_message(submission_author, comment_time, 'submission')
        create_features.units.loc[index, 'submitter_number_comment']\
            = create_features.number_of_message(submission_author, comment_time, 'comment')
        create_features.units.loc[index, 'submitter_seniority_days'] =\
            create_features.calculate_user_seniority(submission_author)
        create_features.units.loc[index, 'is_first_comment_in_tree'],\
            create_features.units.loc[index, 'number_of_comments_in_tree_by_comment_user'], _, _ = \
            create_features.comment_in_tree(comment_author, comment_time, submission_id)

        # Get the time between the submission and the comment time and the ration between the first comment:
        # print('{}: Get the time between the submission and the comment time and the ration between the first comment'
        #       .format((time.asctime(time.localtime(time.time())))))
        time_to_comment = comment['time_between']
        time_between_messages_hour = math.floor(time_to_comment/3600.0)
        time_between_messages_min = math.floor((time_to_comment - 3600*time_between_messages_hour)/60.0)/100.0
        create_features.units.loc[index, 'time_between_messages'] =\
            time_between_messages_hour + time_between_messages_min
        time_until_first_comment, time_between_comment_first_comment =\
            create_features.time_to_first_comment(submission_id, submission_time, comment_time)
        if time_to_comment > 0:
            create_features.units.loc[index, 'time_ratio'] = time_until_first_comment/time_to_comment
        else:
            create_features.units.loc[index, 'time_ratio'] = 0

        create_features.units.loc[index, 'time_until_first_comment'] = time_until_first_comment
        create_features.units.loc[index, 'time_between_comment_first_comment'] = time_between_comment_first_comment

        # Get the numbers of comments by the submitter
        _, create_features.units.loc[index, 'number_of_comments_in_tree_from_submitter'],\
            number_of_respond_by_submitter, number_of_respond_by_submitter_total =\
            create_features.comment_in_tree(submission_author, comment_time, submission_id, comment_author, True)
        create_features.units.loc[index, 'number_of_respond_by_submitter'],\
            create_features.units.loc[index, 'number_of_respond_by_submitter_total'] \
            = number_of_respond_by_submitter, number_of_respond_by_submitter_total

        # Ratio of comments number:
        # print('{}: Ratio of comments number'.format((time.asctime(time.localtime(time.time())))))
        if submission_num_comments == 0:
            create_features.units.loc[index, 'respond_to_comment_user_all_ratio'] = 0
            create_features.units.loc[index, 'respond_total_ratio'] = 0
        else:
            create_features.units.loc[index, 'respond_to_comment_user_all_ratio'] =\
                number_of_respond_by_submitter / submission_num_comments
            create_features.units.loc[index, 'respond_total_ratio'] =\
                number_of_respond_by_submitter_total / submission_num_comments
        if number_of_respond_by_submitter_total == 0:
            create_features.units.loc[index, 'respond_to_comment_user_responses_ratio'] = 0
        else:
            create_features.units.loc[index, 'respond_to_comment_user_responses_ratio'] =\
                number_of_respond_by_submitter / number_of_respond_by_submitter_total

        # Sentiment analysis:
        # for the comment:
        print('{}: Sentiment analysis'.format((time.asctime(time.localtime(time.time())))))
        comment_sentiment_list = sentiment_analysis(comment_body)
        create_features.units.loc[index, 'nltk_com_sen_pos'], create_features.units.loc[index, 'nltk_com_sen_neg'], \
            create_features.units.loc[index, 'nltk_com_sen_neutral'] = \
            comment_sentiment_list[0], comment_sentiment_list[1], comment_sentiment_list[2]
        # for the submission:
        sub_sentiment_list = sentiment_analysis(submission_body)
        create_features.units.loc[index, 'nltk_sub_sen_pos'], create_features.units.loc[index, 'nltk_sub_sen_neg'],\
            create_features.units.loc[index, 'nltk_sub_sen_neutral'] = \
            sub_sentiment_list[0], sub_sentiment_list[1], sub_sentiment_list[2]
        # for the title
        title_sentiment_list = sentiment_analysis(title)
        create_features.units.loc[index, 'nltk_title_sen_pos'], create_features.units.loc[index, 'nltk_title_sen_neg'], \
            create_features.units.loc[index, 'nltk_title_sen_neutral'] = \
            title_sentiment_list[0], title_sentiment_list[1], title_sentiment_list[2]
        # cosine similarity between submission's sentiment vector and comment sentiment vector:
        sentiment_sub = np.array(sub_sentiment_list).reshape(1, -1)
        sentiment_com = np.array(comment_sentiment_list).reshape(1, -1)
        create_features.units.loc[index, 'nltk_sim_sen'] = cosine_similarity(sentiment_sub, sentiment_com)[0][0]

        # percent of adjective in the comment:
        # print('{}: percent of adjective in the comment'.format((time.asctime(time.localtime(time.time())))))
        create_features.units.loc[index, 'percent_adj'] = percent_of_adj(comment_body)

        new_index += 1

    # export the data to csv file
    all_comments_features.T.to_csv(os.path.join(data_directory, 'features_CMV.csv'), encoding='utf-8')
    print('number_of_treatment_minus_1: ', number_of_treatment_minus_1)


if __name__ == '__main__':
    main()
