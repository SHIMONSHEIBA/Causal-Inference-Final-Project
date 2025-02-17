import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


class Sim:

    def __init__(self, data, units, all_submissions):

        self.data = data
        self.units = units
        self.all_submissions = all_submissions

        self.data_preprocess()
        return

    def data_preprocess(self):

        print("{} :begin data pre process".format(time.asctime(time.localtime(time.time()))))

        # remove unicode char from relevant data columns
        self.all_submissions["submission_title"] = \
            self.all_submissions["submission_title"].apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
        self.all_submissions["submission_body"] = \
            self.all_submissions["submission_body"].apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
        self.all_submissions["submission_body"] = self.all_submissions["submission_body"].str.partition(
            "Hello, users of CMV! This is a footnote from your moderators")[0]
        self.data["comment_body"] = \
            self.data["comment_body"].apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))

        # concat submissions text
        self.all_submissions["submission_title_and_body"] = self.all_submissions["submission_title"] \
                                                            + self.all_submissions["submission_body"]

        print("{} :finish data pre process".format(time.asctime(time.localtime(time.time()))))
        return

    def concat_df_rows(self, comment_created_utc, author, is_submission=False):
        if is_submission:
            text = self.all_submissions.loc[(self.all_submissions['submission_created_utc'] <= comment_created_utc) &
                                   (self.all_submissions['submission_author'] == author)]["submission_title_and_body"]
            text_cat = text.str.cat(sep=' ')
            return text_cat

        text = self.data.loc[(self.data['comment_created_utc'] <= comment_created_utc) &
                               (self.data['comment_author'] == author)]["comment_body"]
        text_cat = text.str.cat(sep=' ')

        return text_cat

    def create_vocab(self):

        # get all comments for vocab
        vocab_c = self.data["comment_body"]
        vocab_s = self.all_submissions["submission_title_and_body"]

        # join two strings of comments and submissions
        vocab_df = pd.concat([vocab_c, vocab_s])
        return vocab_df

    def calc_tf_idf_cos(self):

        # create vocabulary for idf calq
        vocab = self.create_vocab()
        print("{} :begin fitting tfidf".format(time.asctime(time.localtime(time.time()))))
        tfidf_vec_fitted = TfidfVectorizer(stop_words='english', lowercase=True, analyzer='word', norm='l2',
                                           smooth_idf=True, sublinear_tf=False, use_idf=True).fit(vocab)
        print("{} :finish fitting tfidf".format(time.asctime(time.localtime(time.time()))))

        print("starting tf idf similarity for units")
        for index, row in self.units.iterrows():
            # define thresholds and filter variables
            comment_created_utc = row.loc['comment_created_utc']
            comment_author = row.loc['comment_author']
            submission_author = row.loc['submission_author']

            # all text of commenter until comment time
            text_commenter = self.concat_df_rows(comment_created_utc, comment_author)
            text_commenter_submission = self.concat_df_rows(comment_created_utc, comment_author, True)
            text_commenter += text_commenter_submission

            # all text of submissioner until comment time
            text_submissioner = self.concat_df_rows(comment_created_utc, submission_author)
            text_submissioner_submission = self.concat_df_rows(comment_created_utc, submission_author, True)
            text_submissioner += text_submissioner_submission

            text = [text_submissioner, text_commenter]

            tfidf_vec_transformed = tfidf_vec_fitted.transform(text)

            similarity = cosine_similarity(tfidf_vec_transformed[0:], tfidf_vec_transformed[1:])

            self.units.loc[index, "submmiter_commenter_tfidf_cos_sim"] = \
                similarity[0][0]

            if index % 100 == 0:
                print("{} finished similarity for unit no {}".format(time.asctime(time.localtime(time.time())), index))

        # save results with new feature
        comment_sim_df = self.units[["comment_id","submmiter_commenter_tfidf_cos_sim"]]
        comment_sim_df.to_csv("units_with_sim_tfidf_cos.csv")


def main():

    # config path and load data
    base_directory = os.path.abspath(os.curdir)
    results_directory = os.path.join(base_directory, 'importing_change_my_view')

    debug_mode = False
    if debug_mode:
        # original data
        comments = pd.read_csv(os.path.join(results_directory, 'all submissions comments with label_small.csv'))
        submissions = pd.read_csv(os.path.join(results_directory, 'all submissions_small.csv'))
        # prepare data and join
        comments['submission_id'] = comments.submission_id.str.slice(2, -1)
        submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")
        join_result = comments.merge(submissions_drop, on='submission_id', how='inner')
        data = join_result
        units = join_result
    else:
        data = pd.read_csv(os.path.join(results_directory, 'all_data_0304.csv'))
        units = pd.read_csv(os.path.join(results_directory, 'units_0304.csv'))
        submissions = pd.read_csv(os.path.join(results_directory, 'all submissions.csv'))
        submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")

    units["submmiter_commenter_tfidf_cos_sim"] = 0
    sim = Sim(data, units, submissions_drop)
    sim.calc_tf_idf_cos()


if __name__ == '__main__':
    main()
