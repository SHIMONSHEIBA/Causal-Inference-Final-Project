import pandas as pd
import urllib.parse
import urllib.request
import time


class SentimentAnalysis:
    def __init__(self):
        # Load all relevant data
        self.units = pd.read_csv(os.path.join(data_directory, 'units.csv'))  # all the units, with label
        pd.to_numeric(self.units['submission_created_utc'])
        pd.to_numeric(self.units['comment_created_utc'])
        self.units = self.units[['comment_body', 'comment_author', 'submission_author', 'submission_body',
                                 'submission_title', 'comment_id', 'parent_id', 'comment_created_utc',
                                 'submission_created_utc', 'submission_id', 'submission_num_comments',
                                 'time_between']]
        self.units['comment_id'] = self.units.comment_id.str.lstrip("b'")
        self.units['comment_id'] = self.units.comment_id.str.rstrip("'")
        self.units['parent_id'] = self.units.parent_id.str.lstrip("b't_1")
        self.units['parent_id'] = self.units.parent_id.str.lstrip("b't_3")
        self.units['parent_id'] = self.units.parent_id.str.rstrip("'")
        self.all_data = pd.read_csv(os.path.join(data_directory, 'all_data.csv'))
        pd.to_numeric(self.all_data['comment_created_utc'])
        pd.to_numeric(self.all_data['submission_created_utc'])
        self.all_data['parent_id'] = self.all_data.parent_id.str.lstrip("b't_1")
        self.all_data['parent_id'] = self.all_data.parent_id.str.lstrip("b't_3")
        self.all_data['comment_id'] = self.all_data.comment_id.str.lstrip("b'")
        self.all_data['comment_id'] = self.all_data.comment_id.str.rstrip("'")
        self.all_data['parent_id'] = self.all_data.parent_id.str.rstrip("'")
        self.all_data = self.all_data[['submission_id', 'comment_author', 'submission_author', 'comment_id',
                                       'comment_created_utc', 'submission_created_utc', 'parent_id', 'comment_body']]


def sentiment_analysis(text):
    data = urllib.parse.urlencode({"text": text})
    data = data.replace("\n", "")
    data = data.lower()
    data = data.encode('ascii')
    with urllib.request.urlopen("http://text-processing.com/api/sentiment/", data) as f:
        result = f.read().decode('utf-8')
        index_of_neg = result.find('neg')
        index_of_neutral = result.find('neutral')
        index_of_pos = result.find('pos')
        index_of_label = result.find('label')
        neg_prob = float(result[index_of_neg + 6:index_of_neutral - 3])
        neutral_prob = float(result[index_of_neutral + 10:index_of_pos - 3])
        pos_prob = float(result[index_of_pos + 6:index_of_label - 4])
        return [pos_prob, neg_prob, neutral_prob]


def main():
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    create_features = CreateFeatures()
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: all data: {}, units data: {}'.format(create_features.all_data.shape,
                                                            create_features.units.shape))