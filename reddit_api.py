import praw
import time
import csv
import pickle
import logging
import os
from datetime import datetime
import pandas as pd
from collections import defaultdict
import re

# configurate logging
base_directory = os.path.abspath(os.curdir)
log_directory = os.path.join(base_directory, 'logs')
results_directory = os.path.join(base_directory, 'importing_change_my_view')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_importing_change_my_view_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )


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
        with open(os.path.join(results_directory, 'all submissions.csv'), 'a') as file:
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
            with open(os.path.join(results_directory, 'all submissions.csv'), 'a') as file:
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
        with open(os.path.join(results_directory, 'all submissions comments.csv'), 'a') as file:
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
                with open(os.path.join(results_directory, 'all submissions comments.csv'), 'a') as file:
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
        all_submissions_comments = pd.read_csv(os.path.join(results_directory, 'all submissions comments.csv'))
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
        with open(os.path.join(results_directory, 'all deltas.csv'), 'a') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames = ['delta_author', 'delta_title', 'delta_created_utc', 'delta_selftext', 'delta_id',
                          'delta_likes', 'delta_ups', 'delta_downs', 'delta_score', 'delta_name', 'delta_permalink']
            writer.writerow(fieldnames)

        # write all deltas to file
        for delta in self.r_connection.subreddit(delta_log).submissions():
            with open(os.path.join(results_directory, 'all deltas.csv'), 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow([delta.author, delta.title.encode('utf-8'), delta.created_utc,
                                 delta.selftext.encode('utf-8'), delta.id.encode('utf-8'),delta.likes, delta.ups,
                                 delta.downs, delta.score, delta.name, delta.permalink.encode('utf-8')])
            num_of_total_deltas += 1
            print("added delta id : {} of title: {}".format(delta.id, delta.title))
            print("total number of deltas so far is {}".format(num_of_total_deltas))
            logging.info("added delta id : {} of title: {}".format(delta.id, delta.title))
            logging.info("total number of deltas so far is {}".format(num_of_total_deltas))

        # TODO: enable this for final code and not from main (just for clear code)
        # parse delta logs for OP deltas
        # OP_deltas_comments_ids_deltalog = self.parse_op_deltas()

        return

    def parse_op_deltas(self):
        """
        this method parse the text of each delta comment and saves the IDs of the ones given by OP
        :return: OP delta comment ids dict {submission id: [comments id]}
        """
        # TODO: change path & name to dynamic
        deltas = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                "Causal-Inference-Final-Project\\"
                                                "importing_change_my_view\\all deltas.csv", index_col=False)

        OP_deltas_comments_ids_deltalog = defaultdict(list)
        comments_with_delta_ids = list()
        delta_count = 0
        for df_index, row in deltas.iterrows():

            # obtain submissionid
            submission_id_indexes = [m.start() for m in
                                     re.finditer('/comments/', deltas.loc[df_index, "delta_selftext"])]
            if len(submission_id_indexes) < 1:
                print("bug")
            submission_id_text = deltas.loc[df_index, "delta_selftext"][:submission_id_indexes[0] + 20]
            submission_id_text_parsed = submission_id_text.split("/")
            comments_idx = submission_id_text_parsed.index("comments")
            submission_id = submission_id_text_parsed[comments_idx + 1]

            # obtain OP user name
            op_username_indexes = [m.start() for m in
                                   re.finditer('"opUsername":', deltas.loc[df_index, "delta_selftext"])]
            op_username_text = deltas.loc[df_index, "delta_selftext"][op_username_indexes[0]:]
            left_op_username_text = op_username_text.partition("\\n")[0]
            op_username = left_op_username_text.partition('",\\n')[0]
            op_username = op_username.split(": ")[1]
            op_username = op_username.strip(",'")

            # parse from delta text all comments id's that got deltas.
            deltas_indexes = [m.start() for m in re.finditer('"awardedLink":', deltas.loc[df_index, "delta_selftext"])]
            delta_awarding_indexes = [m.start() for m in
                                      re.finditer('"awardingUsername":', deltas.loc[df_index, "delta_selftext"])]

            if len(deltas_indexes) != len(delta_awarding_indexes):
                print("match error")
                break

            for delta_list_idx, delta_index in enumerate(deltas_indexes):

                # get & check if awarding is OP
                awarding_text = deltas.loc[df_index, "delta_selftext"][delta_awarding_indexes[delta_list_idx]:]
                left_awarding_text = awarding_text.partition("\\n")[0]
                awarding_username = left_awarding_text.partition('",\\n')[0]
                awarding_username = awarding_username.split(": ")[1]
                awarding_username = awarding_username.strip(",'")

                if awarding_username != op_username:
                    # print("not OP's delta")
                    continue

                # get awarded delta comment id index
                text = deltas.loc[df_index, "delta_selftext"][delta_index:]
                left_text = text.partition("\\n")[0]
                splited = text.split('/')
                for idx, parts in enumerate(splited):
                    if "cmv_" in parts:
                        comment_id = splited[idx + 1].partition('",\\n')[0]
                        print(comment_id)
                        delta_count += 1
                        print(delta_count)
                        comments_with_delta_ids.append(comment_id)
                        OP_deltas_comments_ids_deltalog[submission_id].append(comment_id)

        comments_with_delta_ids = list(set(comments_with_delta_ids))
        print("num of deltas from deltalog is: {}".format(len(comments_with_delta_ids)))

        # save delta data
        print("save delta log data ")
        logging.info("save delta log data")
        with open('OP_deltas_comments_ids_deltalog.pickle', 'wb') as handle:
            pickle.dump(OP_deltas_comments_ids_deltalog, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('comments_with_delta_ids.pickle', 'wb') as handle:
            pickle.dump(comments_with_delta_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return OP_deltas_comments_ids_deltalog

    def create_label(self, OP_deltas_comments_ids_deltalog, OP_deltas_comments_ids):
        """
        this method creates a label for delta label
        :param OP_deltas_comments_ids_deltalog: dict of submission_id : comment id from deltalog
        :param OP_deltas_comments_ids: dict of submission_id : comment id from manual extraction
        :return:
        """

        # get data
        # TODO: change path & name to dynamic
        comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                  "Causal-Inference-Final-Project\\"
                                                  "importing_change_my_view\\all submissions comments.csv",
                               index_col=False)
        # create label column
        comments["delta"] = 0

        # for each comment in data , check if it's in one of the deltas dict list by it's submissionID
        for index, row in comments.iterrows():

            if type(row.loc['comment_id']) is float:
                # TODO: VALIDATE WHY SO MUCH NAN
                print("not real comment id : {}".format(row.loc['comment_id']))
                continue

            deltalog_comments = list()
            manual_comments = list()

            # get delta comments of this submissionid
            try:
                deltalog_comments = OP_deltas_comments_ids_deltalog[row.loc['submission_id']]
            except ValueError:
                print("no delta comments for submission: {} in deltalog".format(row.loc['submission_id']))

            try:
                manual_comments = OP_deltas_comments_ids[row.loc['submission_id']]
                manual_comments = [w.lstrip("b") for w in manual_comments]
                manual_comments = [w.strip("'") for w in manual_comments]
                manual_comments = [w.lstrip("t1_") for w in manual_comments]

            except ValueError:
                print("no delta comments for submission: {} in manual".format(row.loc['submission_id']))

            # check if this comment got delta
            if row.loc['comment_id'].lstrip("b").strip("'") in deltalog_comments \
                    or row.loc['comment_id'].lstrip("b").strip("'") in manual_comments:
                comments.loc[index, "delta"] = 1

        # see label distribution in data
        print("VALUE COUNT:")
        print(comments['delta'].value_counts())

        # validate that total number of deltas equal to the label distribution above
        count_deltas = list()
        for key, value in OP_deltas_comments_ids_deltalog.items():
            count_deltas += value
        print("num of deltas in manual: {}".format(len(count_deltas)))
        for k, value in OP_deltas_comments_ids.items():
            count_deltas += value
        count_deltas = list(set(count_deltas))
        print("TOTAL {} deltas".format(len(count_deltas)))

        # save data with label
        #TODO: change path & name to dynamic
        comments.to_csv(path_or_buf="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
                                                  "Causal-Inference-Final-Project\\"
                                                  "importing_change_my_view\\all submissions comments with label.csv")
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

    # parse delta logs for OP deltas
    OP_deltas_comments_ids_deltalog = connect.parse_op_deltas()

    # TODO: change save/load places to self class variables
    pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
    OP_deltas_comments_ids = pickle.load(pkl_file)

    connect.create_label(OP_deltas_comments_ids_deltalog, OP_deltas_comments_ids)


if __name__ == '__main__':
    main()
