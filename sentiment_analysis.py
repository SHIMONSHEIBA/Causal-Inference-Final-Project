import pandas as pd
import urllib.parse
import urllib.request
import time
import os
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk as nk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")


class SentimentAnalysis:
    def __init__(self):
        # Load all relevant data
        self.units = pd.read_csv(os.path.join(data_directory, 'units_0304.csv'))  # all the units, with label
        self.units = self.units[['comment_body', 'submission_body', 'submission_title', 'comment_id']]
        self.units['comment_id'] = self.units.comment_id.str.lstrip("b'")
        self.units['comment_id'] = self.units.comment_id.str.rstrip("'")

        self.units.assign(nltk_com_sen_pos='', nltk_com_sen_neg='', nltk_com_sen_neutral='', nltk_sub_sen_pos='',
                          nltk_sub_sen_neg='', nltk_sub_sen_neutral='', nltk_title_sen_pos='',
                          nltk_title_sen_neg='', nltk_title_sen_neutral='', nltk_sim_sen='', percent_adj='')


def sentiment_analysis(text):
    # data = urllib.parse.urlencode({"text": text})
    # data = data.replace("\n", "")
    # data = data.lower()
    # data = data.encode('ascii')
    # with urllib.request.urlopen("http://text-processing.com/api/sentiment/", data) as f:
    #     result = f.read().decode('utf-8')
    #     index_of_neg = result.find('neg')
    #     index_of_neutral = result.find('neutral')
    #     index_of_pos = result.find('pos')
    #     index_of_label = result.find('label')
    #     neg_prob = float(result[index_of_neg + 6:index_of_neutral - 3])
    #     neutral_prob = float(result[index_of_neutral + 10:index_of_pos - 3])
    #     pos_prob = float(result[index_of_pos + 6:index_of_label - 4])
    sid = SentimentIntensityAnalyzer()
    result = sid.polarity_scores(text)
    neg_prob = result['neg']
    neutral_prob = result['neu']
    pos_prob = result['pos']
    return [pos_prob, neg_prob, neutral_prob]


def get_POS(text):
    text_parsed = nk.word_tokenize(text)
    words_pos = nk.pos_tag(text_parsed)

    return words_pos


def percent_of_adj(text):
    pos_text = get_POS(text)
    pos_df = pd.DataFrame(pos_text, columns=['word', 'POS'])
    number_all_pos = pos_df.shape[0]
    all_pos = pos_df['POS']
    freq = nk.FreqDist(all_pos)
    number_adj_pos = freq['JJ'] + freq['JJS'] + freq['JJR']
    percent_of_adj_pos = number_adj_pos/number_all_pos

    return percent_of_adj_pos


def main():
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    create_features = SentimentAnalysis()
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: units data: {}'.format(create_features.units.shape))

    all_comments_features = pd.DataFrame()
    new_index = 0
    for index, comment in create_features.units.iterrows():
        if new_index % 100 == 0:
            print('{}: Finish calculate {} samples'.format((time.asctime(time.localtime(time.time()))), new_index))

        comment_body = copy(comment['comment_body'])
        submission_body = copy(comment['submission_body'])
        title = copy(comment['submission_title'])

        # Sentiment analysis:
        # for the comment:
        comment_sentiment_list = sentiment_analysis(comment_body)
        comment['nltk_com_sen_pos'], comment['nltk_com_sen_neg'], comment['nltk_com_sen_neutral'] = \
            comment_sentiment_list[0], comment_sentiment_list[1], comment_sentiment_list[2]
        # for the submission:
        sub_sentiment_list = sentiment_analysis(submission_body)
        comment['nltk_sub_sen_pos'], comment['nltk_sub_sen_neg'], comment['nltk_sub_sen_neutral'] = \
            sub_sentiment_list[0], sub_sentiment_list[1], sub_sentiment_list[2]
        # for the title
        title_sentiment_list = sentiment_analysis(title)
        comment['nltk_title_sen_pos'], comment['nltk_title_sen_neg'], comment['nltk_title_sen_neutral'] = \
            title_sentiment_list[0], title_sentiment_list[1], title_sentiment_list[2]
        # cosine similarity between submission's sentiment vector and comment sentiment vector:
        sentiment_sub = np.array(sub_sentiment_list).reshape(1, -1)
        sentiment_com = np.array(comment_sentiment_list).reshape(1, -1)
        comment['nltk_sim_sen'] = cosine_similarity(sentiment_sub, sentiment_com)[0][0]

        # percent of adjective in the comment:
        comment['percent_adj'] = percent_of_adj(comment_body)

        all_comments_features = pd.concat([all_comments_features, comment], axis=1)
        # all_comments_features.T.to_csv(os.path.join(data_directory, 'sentiment_analysis_CMV.csv'), encoding='utf-8')

        new_index += 1

    # export the data to csv file
    all_comments_features.T.to_csv(os.path.join(data_directory, 'sentiment_analysis_CMV.csv'), encoding='utf-8')


if __name__ == '__main__':
    main()
