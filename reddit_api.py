import praw
import time
import csv
import pickle
import logging
import os
from datetime import datetime
import pandas as pd
from collections import defaultdict

class ApiConnection:

    def __init__(self, subreddit):

        # configurate API connection details
        user_agent = "learning API 1.0"
        client_id = 'learning API 1.0'
        user = 'ssheiba'
        password = 'Angui100'
        ssheiba_client_id = 'ysLnQ_KXpkYA1g'
        ssheiba_client_secret = 'pLMhOt7sVy6IHdx5mcCy48yB6Ow'
        ssheiba_user_agent = 'learningapp:com.example.myredditapp:v1.2.3 (by /u/ssheiba)'
        self.r_connection = praw.Reddit(client_id=ssheiba_client_id, user_agent=ssheiba_user_agent,
                                        client_secret=ssheiba_client_secret, username=user,password=password)
        self.subreddit_name = subreddit

        # configurate logging
        base_directory = os.path.abspath(os.curdir)
        self.log_directory = os.path.join(base_directory, 'logs')
        self.results_directory = os.path.join(base_directory, 'importing_change_my_view')
        LOG_FILENAME = datetime.now().strftime(os.path.join(self.log_directory,
                                                            datetime.now().strftime
                                                            ('LogFile_importing_change_my_view_%d_%m_%Y_%H_%M_%S.log')))
        logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

        return

    def get_karma(self, user_id):
        user = self.r_connection.redditor(user_id)
        comments = user.comments
        gen = user.get_submitted(limit=10)
        karma_by_subreddit = {}
        for thing in gen:
            subreddit = thing.subreddit.display_name
            karma_by_subreddit[subreddit] = (karma_by_subreddit.get(subreddit, 0)
                                             + thing.score)

    def get_submissions(self):

        num_of_total_submissions = 0
        submissions = list()
        with open(os.path.join(self.results_directory, 'all submissions.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['submission_author', 'submission_title', 'submission_comments_by_id',
                          'submission_created_utc', 'submission_edited', 'submission_body', 'submission_id',
                          'submission_likes', 'submission_ups', 'submission_downs', 'submission_score',
                          'submission_num_reports', 'submission_gilded', 'submission_distinguished',
                          'submission_is_crosspostable', 'submission_banned_by', 'submission_banned_at_utc',
                          'submission_removal_reason', 'submission_clicked', 'submission_num_comments',
                          'submission_contest_mode', 'submission_media']
            writer.writerow(fieldnames)
        subids = set()
        for submission in self.r_connection.subreddit(self.subreddit_name).submissions():
            with open(os.path.join(self.results_directory, 'all submissions.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([submission.author, submission.title.encode('utf-8'),
                                 submission._comments_by_id, submission.created_utc,
                                 submission.edited, submission.selftext.encode('utf-8'), submission.id,
                                 submission.likes, submission.ups, submission.downs, submission.score,
                                 submission.num_reports, submission.gilded, submission.distinguished,
                                 submission.is_crosspostable, submission.banned_by, submission.banned_at_utc,
                                 submission.removal_reason, submission.clicked, submission.num_comments,
                                 submission.contest_mode, submission.media])

            subids.add(submission.id)
            submissions.append(submission)
            num_of_total_submissions += 1
            print("added submission id : {}".format(submission.id))
            print("total number of submissions so far is {}".format(num_of_total_submissions))
            logging.info("added submission id : {}".format(submission.id))
            logging.info("total number of submissions so far is {}".format(num_of_total_submissions))
        subid = list(subids)
        print("all subids are: {}".format(subid))
        logging.info("all subids are: {}".format(subid))
        # save all submissions object
        print("saving submissions list")
        logging.info("saving submissions list")
        with open('submissions.pickle', 'wb') as handle:
            pickle.dump(submissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return subid

    def parse_comments(self, subid):
        """
        this method retrieves all comments for the subid list (and their replies)
        :param subid: list of submissions id to get their comment tree
        :return: saves features of the comments and replies, among with the data objects of them.
        """
        comments = dict()
        num_of_total_comments = 0
        index = len(subid)

        # prepare the file
        with open(os.path.join(self.results_directory, 'all submissions comments.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames2 = ['comment_author', 'comment_created_utc', 'comment_edited', 'comment_body',
                           'comment_path', 'comment_id', 'parent_id', 'submission_id', 'comment_is_submitter',
                           'comment_likes', 'comment_is_root', 'comment_ups', 'comment_downs', 'comment_score',
                           'comment_num_reports', 'comment_gilded', 'comment_distinguished', 'comment_controversiality',
                           'comment_banned_by', 'comment_banned_at_utc', 'comment_depth', 'comment_removal_reason']
            writer.writerow(fieldnames2)

        # iterate all submissions
        for i in range(0, index):
            print('{}: start submission {} for id {}'.format((time.asctime(time.localtime(time.time()))), i, subid[i]))
            logging.info('{}: start submission {} for id {}'.format((time.asctime(time.localtime(time.time()))), i,
                                                                    subid[i]))
            submission = self.r_connection.submission(id=subid[i])
            print('{}: start more comments'.format((time.asctime(time.localtime(time.time())))))
            logging.info('{}: start more comments'.format((time.asctime(time.localtime(time.time())))))

            # replace the more comments objects with the comments themselves (flatten trees of replies of comments)
            submission.comment_sort = 'new'
            submission.comments.replace_more(limit=None)

            comments[subid[i]] = list()

            # retrieve all comments data
            for comment in submission.comments.list():

                comments[subid[i]].append(comment)
                with open(os.path.join(self.results_directory, 'all submissions comments.csv'), 'a') as file:
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow([comment.author, comment.created_utc,
                                     comment.edited,
                                     comment.body.encode('utf-8'), comment.permalink.encode('utf-8'),
                                     comment.id.encode('utf-8'), comment.parent_id.encode('utf-8'),
                                     comment.submission.id.encode('utf-8'),comment.is_submitter,
                                     comment.likes, comment.is_root, comment.ups, comment.downs, comment.score,
                                     comment.num_reports, comment.gilded, comment.distinguished,
                                     comment.controversiality, comment.banned_by, comment.banned_at_utc, comment.depth,
                                     comment.removal_reason])
            print("number of comments for subid {} is {}".format(subid[i], len(comments[subid[i]])))
            logging.info("number of comments for subid {} is {}".format(subid[i], len(comments[subid[i]])))
            num_of_total_comments += len(comments[subid[i]])
            print("total number of comments so far is {}".format(num_of_total_comments))
            logging.info("total number of comments so far is {}".format(num_of_total_comments))

        # extract deltas from comments
        all_submissions_comments = pd.read_csv(os.path.join(self.results_directory, 'all submissions comments.csv'))
        self.get_deltas_manual(all_submissions_comments)

        # save all comments object
        # with open('comments.pickle', 'wb') as handle:
        #     pickle.dump(comments, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def get_deltas_manual(self, all_submissions_comments):

        OP_deltas_comments_ids = defaultdict(list)
        delta_tokens = ['Δ', '!delta', '∆', '&#8710;']
        num_of_deltas = 0
        OP_deltas = defaultdict(dict)

        # find all delta comments and save their details in OP_deltas_comments_ids and the comments that got the delta
        # ID in OP_deltas
        for index, row in all_submissions_comments.iterrows():
            if row.loc['comment_is_submitter'] == True and (delta_tokens[0] in row.loc['comment_body'] or
                                                                    delta_tokens[1] in row.loc['comment_body'] or
                                                                    delta_tokens[2] in row.loc['comment_body'] or
                                                                    delta_tokens[3] in row.loc['comment_body']) \
                    and len(row.loc['comment_body']) > 50:
                num_of_deltas += 1
                print("{} deltas".format(num_of_deltas))
                OP_deltas_comments_ids[row.submission_id].append(row.parent_id)
                OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_OP"] = row.comment_author
                OP_deltas[(row.submission_id, row.parent_id)][
                    row.comment_id + "_" + "delta_date"] = row.comment_created_utc
                OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_body"] = row.comment_body
                OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_path"] = row.comment_path
                OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_id"] = row.comment_id
                OP_deltas[(row.submission_id, row.parent_id)][row.comment_id + "_" + "delta_parent_id"] = row.parent_id
                OP_deltas[(row.submission_id, row.parent_id)][
                    row.comment_id + "_" + "delta_submission_id"] = row.submission_id

        # save delta data
        print("save delta data")
        logging.info("save delta data")
        with open('OP_deltas_comments_ids.pickle', 'wb') as handle:
            pickle.dump(OP_deltas_comments_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('OP_deltas.pickle', 'wb') as handle:
            pickle.dump(OP_deltas, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def get_deltas_log(self, delta_log):
        """
        this method retrieves all deltas awarded in CMV and saves that log to a csv.
        :param delta_log: name of subreddit that holds all the deltas given in CMV subreddit
        :return:
        """
        num_of_total_deltas = 0

        # create file
        with open(os.path.join(self.results_directory, 'all deltas.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['delta_author', 'delta_title', 'delta_created_utc', 'delta_selftext', 'delta_id',
                          'delta_likes', 'delta_ups', 'delta_downs', 'delta_score', 'delta_name', 'delta_permalink']
            writer.writerow(fieldnames)

        # write all deltas to file
        for delta in self.r_connection.subreddit(delta_log).submissions():
            with open(os.path.join(self.results_directory, 'all deltas.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([delta.author, delta.title.encode('utf-8'), delta.created_utc,
                                 delta.selftext.encode('utf-8'), delta.id.encode('utf-8'),delta.likes, delta.ups,
                                 delta.downs, delta.score, delta.name, delta.permalink.encode('utf-8')])
            num_of_total_deltas += 1
            print("added delta id : {} of title: {}".format(delta.id, delta.title))
            print("total number of deltas so far is {}".format(num_of_total_deltas))
            logging.info("added delta id : {} of title: {}".format(delta.id, delta.title))
            logging.info("total number of deltas so far is {}".format(num_of_total_deltas))

        return


def main():

    subreddit = 'changemyview'
    print('{} : Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    logging.info('{} : Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    # create class instance
    connect = ApiConnection(subreddit)

    # get submissions of sub reddit
    subids = connect.get_submissions()

    # get comments of submissions
    connect.parse_comments(subids)

    print('{} : finished Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))
    logging.info('{} : finished Run for sub reddit {}'.format((time.asctime(time.localtime(time.time()))), subreddit))

    # get outcome from delta log
    delta_log = 'DeltaLog'
    connect.get_deltas_log(delta_log)

if __name__ == '__main__':
    main()
