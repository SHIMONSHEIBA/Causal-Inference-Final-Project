
#first run this, download() will open an installation gui

# nltk.download()
import pandas as pd
# after installation you will be able to import this:
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from collections import defaultdict
import nltk as nk
import os
import pickle



class Sentiment():

    def __init__(self, data_name):

        self.data_name = data_name
        self.base_directory = os.path.abspath(os.curdir)
        self.data_loc = os.path.join(self.base_directory, self.data_name)
        self.data = pd.DataFrame()
        self.senti_dict = defaultdict(dict)
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

    def calc_comments_probs(self, thresh):

        comment_index = 0
        g = lambda x,y: True if x > y else False

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
            comment_pos_neg_rules_df = dict()
            for thr in thresh:
                comment_pos_neg_rules_df[str(thr)] = pd.DataFrame(columns=['pos', 'neg', 'obj'])

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

                    # check for rules of pos/neg labels
                    word_pos_neg_rules_vec = dict()
                    for thr in thresh:
                        word_pos_neg_rules_vec[str(thr)] = self.pos_neg_rules(thr , word_prob_vc)
                        comment_pos_neg_rules_df[str(thr)] = comment_pos_neg_rules_df[str(thr)].\
                            append(word_pos_neg_rules_vec[str(thr)])
                    calculated_words_count += 1
                total_words_count += 1
            self.senti_dict[comment_index]['comment_parsed'] = comment_parsed
            self.senti_dict[comment_index]['total_words_count'] = total_words_count
            self.senti_dict[comment_index]['calculated_words_count'] = calculated_words_count
            self.senti_dict[comment_index]['comment_prob_df_mean'] = comment_prob_df.mean(axis=0)
            self.senti_dict[comment_index]['comment_senti'] = comment_prob_df.mean(axis=0).idxmax(axis=1)
            for thr in thresh:
                self.senti_dict[comment_index]['comment_'+str(thr)+'_cnt'] = comment_pos_neg_rules_df[str(thr)].sum()
                self.senti_dict[comment_index]['comment_pos_greater_'+str(thr)] = \
                    g(comment_prob_df[comment_prob_df['pos'] > 0.5]['pos'].mean(axis=0),
                                                                                    comment_prob_df[
                                                                                        comment_prob_df['neg'] > 0.5][
                                                                                        'neg'].mean(axis=0))


            # print('comment probs are: {}'.format(comment_prob_df.mean(axis=0)))
            # print('total words in comment : {}'.format(total_words_count))
            print('for comment {} calculated words: {}, meaning {} of the post words'.format(comment_index,
                                                                                             calculated_words_count,
                                                                              "{:.1%}".format(calculated_words_count/
                                                                                             total_words_count)))
            comment_index+=1

        print('saving senti dict')
        util.save_dict(self.base_directory, ' senti dict_short', self.senti_dict)

        return

    def senti_analyze_dict(self, dict_path, dict_name, thresh, word_num):

        print('analyzing dict {}'.format(dict_path + dict_name))

        pkl_file = open(dict_path + dict_name +'.pkl', 'rb')
        senti_dict = pickle.load(pkl_file)
        pkl_file.close()

        neg_cnt = 0
        obj_cnt = 0
        pos_cnt = 0
        comment_cnt = 0
        thresh_dict = defaultdict(dict)
        for thr in thresh:
            thresh_dict[str(thr)]['pos']=0
            thresh_dict[str(thr)]['neg'] = 0
            thresh_dict[str(thr)]['obj'] = 0


        for key, value in senti_dict.items():
            if senti_dict[key]['comment_senti'] == 'pos':
                pos_cnt +=1
            elif senti_dict[key]['comment_senti'] == 'neg':
                neg_cnt += 1
            else:
                obj_cnt +=1

            for thr in thresh:
                if senti_dict[key]['comment_pos_greater_' + str(thr)]:
                    if senti_dict[key]['comment_'+str(thr)+'_cnt']['pos'] > word_num:
                        thresh_dict[str(thr)]['pos']+=1
                    else:
                        thresh_dict[str(thr)]['obj'] += 1
                else:
                    if senti_dict[key]['comment_prob_df_mean']['neg'] > senti_dict[key]['comment_prob_df_mean']['pos']:
                        if senti_dict[key]['comment_'+str(thr)+'_cnt']['neg'] > word_num:
                            thresh_dict[str(thr)]['neg'] += 1
                        else:
                            thresh_dict[str(thr)]['obj'] += 1
                    else:
                        if senti_dict[key]['comment_prob_df_mean']['neg'] == 0:
                            thresh_dict[str(thr)]['obj'] += 1
                        else:
                            #TODO: how many words passed thr 






            comment_cnt +=1

        print('out of {} comments, pos: {}, neg: {}, obj: {}'.format(comment_cnt,pos_cnt,neg_cnt,obj_cnt))
        return

    def pos_neg_rules(self, thresh, word_prob_vc):

        f = lambda x: True if x > thresh else False
        pos_neg_rules_vec = pd.DataFrame.from_records([{'pos': f(word_prob_vc.iloc[0, 2]),
                                                        'neg': f(word_prob_vc.iloc[0, 0]),
                                                        'obj': f(word_prob_vc.iloc[0, 1])}])

        return pos_neg_rules_vec


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

        output = open(dict_path + dict_name + '.pkl', 'wb')
        pickle.dump(dict, output)
        output.close()

        print('finished saving dict')


def main():
    base_directory = os.path.abspath(os.curdir)
    data_name = 'predictedResultsEfficiency_final_short.xlsx'
    dict_name = ' senti dict_short'

    # create sentiment object class
    sentiment = Sentiment(data_name)

    #classify comments
    pos_neg_thresh = [0.65,0.7,0.8]
    word_num = 3
    sentiment.calc_comments_probs(pos_neg_thresh)

    # analyze sentiment histogram
    sentiment.senti_analyze_dict(base_directory, dict_name, pos_neg_thresh, word_num)


if __name__ == '__main__':
    main()
