
#first run this, download() will open an installation gui

# nltk.download()
import pandas as pd
# after installation you will be able to import this:
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from collections import defaultdict
import nltk as nk
import os
import csv
from os.path import join


class Sentiment():

    def __init__(self, data_name):

        self.data_name = data_name
        self.base_directory = os.path.abspath(os.curdir)
        self.data_loc = os.path.join(self.base_directory, self.data_name)
        self.data = pd.DataFrame()
        self.senti_dict = defaultdict(list)
        self.load_data()

    def load_data(self):

        self.data = pd.read_excel(self.data_loc)
        print('data shape is: {}'.format(self.data.shape))
        print('data columns are {}'.format(self.data.columns))

        return

    def parse_comment(self, comment):

        # get POS
        comment_parsed = nk.word_tokenize(comment)
        # print('parsed comment is: {}'.format(comment_parsed))

        return comment_parsed

    def get_POS(self, comment_parsed):

        words_pos = nk.pos_tag(comment_parsed)

        return words_pos

    def calc_comments_probs(self):

        comment_index = 0
        for comment in self.data.comment_body:

            if not type(comment) is str:
                continue

            if len(comment) == 0:
                continue

            total_words_count = 0
            calculated_words_count = 0

            comment_parsed = self.parse_comment(comment)
            words_pos = self.get_POS(comment_parsed)
            comment_prob_df = pd.DataFrame(columns=['pos','neg','obj'])

            for word in comment_parsed:
                word_pos = words_pos[total_words_count][1]
                # change POS to wordnet POS
                #POS_LIST =  nouns, verbs, adjectives and adverbs = n,v,a,r
                wn_word_POS = util.get_wordnet_pos(word_pos)
                #get synset of chosen POS
                wordnet = swn.senti_synsets(word, wn_word_POS)
                if len(list(wordnet)) > 0:
                    wordnet = swn.senti_synsets(word, wn_word_POS)
                    wordnet0 = list(wordnet)[0]
                    word_prob_vc = pd.DataFrame.from_records([{'pos': wordnet0.pos_score(), 'neg': wordnet0.neg_score(),
                                                               'obj': wordnet0.obj_score()}])
                    comment_prob_df = comment_prob_df.append(word_prob_vc)
                    # print('probs for word {} are: {} '.format(word, word_prob_vc))
                    calculated_words_count += 1
                total_words_count += 1
            self.senti_dict[comment_index] = {'comment_parsed': comment_parsed, 'total_words_count': total_words_count,
                                                  'calculated_words_count': calculated_words_count,
                                                  'comment_prob_df_mean': comment_prob_df.mean(axis=0),
                                              'comment_senti': comment_prob_df.mean(axis=0).idxmax(axis=1)}


            # print('comment probs are: {}'.format(comment_prob_df.mean(axis=0)))
            # print('total words in comment : {}'.format(total_words_count))
            print('for comment {} calculated words: {}, meaning {} of the post words'.format(comment_index,
                                                                                             calculated_words_count,
                                                                              "{:.1%}".format(calculated_words_count/
                                                                                             total_words_count)))
            comment_index+=1

        print('saving senti dict')
        util.save_dict(self.base_directory, 'senti dict', self.senti_dict)

        return



class util():

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return ''

    @staticmethod
    def save_dict(dict_path, dict_name, dict):
        w = csv.writer(open(dict_path + dict_name + '.csv', 'w'))
        for key, val in dict.items():
            w.writerow([key, val])
        print('finished saving dict')


def main():
    data_name = 'predictedResultsEfficiency_final.xlsx'
    sentiment = Sentiment(data_name)
    sentiment.calc_comments_probs()


if __name__ == '__main__':
    main()
