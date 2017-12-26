import pandas as pd
import nltk
import math
import numpy as np


class LanguageModel:

    def __init__(self, is_efficient, data_frame):
        if is_efficient:
            self.relevant_referrals = data_frame.loc[data_frame.IsEfficient == 1]
            self.other_ref = data_frame.loc[data_frame.IsEfficient == -1]
        else:
            self.relevant_referrals = data_frame.loc[data_frame.IsEfficient == -1]
            self.other_ref = data_frame.loc[data_frame.IsEfficient == 1]

        # work on efficient referrals:
        self.relevant_referrals_text = pd.Series(self.relevant_referrals['comment_body'].values)
        self.other_ref_text = pd.Series(self.other_ref['comment_body'].values)

        # concat all the comments
        self.relevant_referrals_all_text = self.relevant_referrals_text.str.cat(sep=' ')

        # remove /n and other signs from the string
        # TODO: we can look for more cases, but I think that for the start it's enough
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace('\n', '')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace('.', '')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace(',', '')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace(':', '')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace('(', '')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace(')', '')
        # self.relevant_referrals_all_text = self.relevant_referrals_all_text.replace('/', ' ')
        self.relevant_referrals_all_text = self.relevant_referrals_all_text.lower()

        # create a list of all the words in the all the comments
        self.relevant_referrals_all_text_list = self.relevant_referrals_all_text.split(' ')
        #self.relevant_referrals_all_text_list = self.relevant_referrals_all_text_list.remove('')

        # len of all comments in each group
        self.relevant_referrals_len = len(self.relevant_referrals_all_text_list)

        # create 1gram FreqDist (kind of dictionary)
        self.relevant_referrals_freq_1gram = nltk.FreqDist(self.relevant_referrals_all_text_list)

        # create 2gram FreqDist (kind of dictionary)
        self.relevant_referrals_freq_2gram = nltk.ConditionalFreqDist(
            nltk.bigrams(self.relevant_referrals_all_text_list))

        # create 2gram condition probability: maps each pair of words to probability
        self.relevant_referrals_prob_2gram = nltk.ConditionalProbDist(self.relevant_referrals_freq_2gram,
                                                                      nltk.MLEProbDist)

    def unigram_prob(self, word):
        return self.relevant_referrals_freq_1gram[word] / self.relevant_referrals_len


    def calc_referrals_prob(self):
        #TODO: CHECK WHY relevant_referrals_1gram_prob_calc LEN IS 428 BUT relevant_referrals_text LEN IS 446
        # TODO: CHANGE TO LOG SCALE
        # TODO: ADD BIGRAM

        self.relevant_referrals_1gram_prob_calc = {}
        self.other_ref_text_1gram_prob_calc = {}

        for ref in self.relevant_referrals_text:
            ref = ref.rstrip('\n')
            words_list = ref.split(' ')
            prob = 1
            for word in words_list:
                # TODO: ADD HANDLING WHEN WHEN WORD NOT IN..
                if word in self.relevant_referrals_freq_1gram.keys():
                    #prob += math.log(self.unigram_prob(word))
                    prob *= self.unigram_prob(word)
            self.relevant_referrals_1gram_prob_calc[ref] = prob

        for ref in self.other_ref_text:
            ref = ref.rstrip('\n')
            words_list = ref.split(' ')
            prob = 1
            for word in words_list:
                if word in self.relevant_referrals_freq_1gram.keys():
                    # prob += math.log(self.unigram_prob(word))
                    prob *= self.unigram_prob(word)
            self.other_ref_text_1gram_prob_calc[ref] = prob

        return

def compare_prob(efficient_language_model_object, non_efficient_language_model_object):

    efficient_ref_ratio = {}
    efficient_ref_ratio_list = []
    non_efficient_ref_ratio = {}
    non_efficient_ref_ratio_list = []
    for key, value in efficient_language_model_object.relevant_referrals_1gram_prob_calc.items():
        if non_efficient_language_model_object.other_ref_text_1gram_prob_calc[key] == 0:
            efficient_ref_ratio[key] = 2
            efficient_ref_ratio_list.append(2)
            continue
        if value == 0:
            efficient_ref_ratio[key] = 0
            efficient_ref_ratio_list.append(0)
            continue
        efficient_ref_ratio[key] = (value/non_efficient_language_model_object.other_ref_text_1gram_prob_calc[key])
        efficient_ref_ratio_list.append(value/non_efficient_language_model_object.other_ref_text_1gram_prob_calc[key])

    for key, value in non_efficient_language_model_object.relevant_referrals_1gram_prob_calc.items():
        if efficient_language_model_object.other_ref_text_1gram_prob_calc[key] == 0:
            non_efficient_ref_ratio[key] = 2
            non_efficient_ref_ratio_list.append(2)
            continue
        if value == 0:
            non_efficient_ref_ratio[key] = 0
            non_efficient_ref_ratio_list.append(0)
            continue
        non_efficient_ref_ratio[key] = (value/efficient_language_model_object.other_ref_text_1gram_prob_calc[key])
        non_efficient_ref_ratio_list.append(value/efficient_language_model_object.other_ref_text_1gram_prob_calc[key])

    print("efficient referrals ratio mean is: {}".format(np.mean(efficient_ref_ratio_list)))
    print("efficient referrals ratio median is: {}".format(np.median(efficient_ref_ratio_list)))
    print("efficient referrals ratio std is: {}".format(np.std(efficient_ref_ratio_list)))
    print("non efficient referrals ratio mean is: {}".format(np.mean(non_efficient_ref_ratio_list)))
    print("non efficient referrals ratio median is: {}".format(np.median(non_efficient_ref_ratio_list)))
    print("non efficient referrals ratio std is: {}".format(np.std(non_efficient_ref_ratio_list)))

    return efficient_ref_ratio, non_efficient_ref_ratio


def main():
    # upload the data ad pandas DF
    all_refarrals = pd.read_excel('FinalFeatures_with_comment_time.xlsx')
    # take relevant columns for the language model
    all_refarrals = all_refarrals[['comment_body', 'IsEfficient']]

    efficient_language_model_object = LanguageModel(is_efficient=True, data_frame=all_refarrals)
    efficient_language_model_object.calc_referrals_prob()
    non_efficient_language_model_object = LanguageModel(is_efficient=False, data_frame=all_refarrals)
    non_efficient_language_model_object.calc_referrals_prob()
    efficient_ref_ratio, non_efficient_ref_ratio = compare_prob(efficient_language_model_object, non_efficient_language_model_object)



if __name__ == '__main__':
    main()

