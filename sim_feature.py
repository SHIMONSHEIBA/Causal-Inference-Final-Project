import csv
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Sim:

    def __init__(self, join_result):

        self.join_result = join_result

        return

    def concat_df_rows(self, comment_created_utc, author, is_submission=False):
        if is_submission:
            text = self.join_result.loc[(self.join_result['comment_created_utc'] < comment_created_utc) &
                                   (self.join_result['comment_author'] == author)].iloc[0][["submission_title",
                                                                                       "submission_body"]]
            text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
            text["submission_body"] = text["submission_body"].partition(
                "Hello, users of CMV! This is a footnote from your moderators")[0]
            text["submission_title_and_body"] = text["submission_title"] + text["submission_body"]
            text_cat = text.str.cat(sep=' ')
            return text_cat

        text = self.join_result.loc[(self.join_result['comment_created_utc'] < comment_created_utc) &
                               (self.join_result['comment_author'] == author)]["comment_body"]
        text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
        text_cat = text.str.cat(sep=' ')

        return text_cat

    def calc_tf_idf_cos(self):

        for index, row in self.join_result.iterrows():
            # define thresholds anf filter variables
            comment_created_utc = row.loc['comment_created_utc']
            comment_author = row.loc['comment_author']
            submission_author = row.loc['submission_author']

            # all text of commenter until comment time
            text_commenter = self.concat_df_rows(comment_created_utc, comment_author)

            # all text of submissioner until comment time
            text_submissioner = self.concat_df_rows(comment_created_utc, submission_author)
            text_submissioner_submission = self.concat_df_rows(comment_created_utc, submission_author, True)
            text_submissioner += text_submissioner_submission

            text = [text_submissioner, text_commenter]
            tfidf = TfidfVectorizer(stop_words='english', lowercase=True, analyzer='word', norm='l2',
                                    smooth_idf=True, sublinear_tf=False, use_idf=True).fit_transform(text)
            similarity = cosine_similarity(tfidf[0:], tfidf[1:])

            self.join_result["submmiter_commenter_tfidf_cos_sim"].loc[index, "submmiter_commenter_tfidf_cos_sim"] = \
                similarity[0][0]

            if similarity[0][0] > 0.9 or similarity[0][0] < 0.2:
                print(similarity[0][0])
                print("text submissioner:")
                print(text_submissioner)
                print("text commenter:")
                print(text_commenter)

        # save results with new feature
        self.join_result.to_csv("join_result_with_sim_tfidf_cos")


def main():

    # TODO: debug when submission and comment are exactly same text
    # config path and load data
    base_directory = os.path.abspath(os.curdir)
    results_directory = os.path.join(base_directory, 'importing_change_my_view')
    comments = pd.read_csv(os.path.join(results_directory, 'all submissions comments with label.csv'))
    submissions = pd.read_csv(os.path.join(results_directory, 'all submissions.csv'))

    # prepare data and join
    comments['submission_id'] = comments.submission_id.str.slice(2, -1)
    submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")
    join_result = comments.merge(submissions_drop, on='submission_id', how='inner')
    join_result["submmiter_commenter_tfidf_cos_sim"] = 0

    sim = Sim(join_result)
    sim.calc_tf_idf_cos()


if __name__ == '__main__':
    main()
