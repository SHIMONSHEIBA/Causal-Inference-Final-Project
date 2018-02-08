
#first run this, download() will open an installation gui
import nltk as nk
# nltk.download()
import pandas as pd
# after installation you will be able to import this:
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from collections import defaultdict

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# data = pd.DataFrame()
# data.from_csv('FinalFeatures_with_comment_time.csv', encoding='utf-8')
# print(data.shape)

comment = 'check out peoples instagram accounts for new ideas; for example, look at the beautiful and low calorie dishes on this persons feed'
comment_parsed = comment.split(' ')
print(comment_parsed)

# get POS
text = nk.word_tokenize(comment)
words_POS = nk.pos_tag(text)

total_words_count = 0
calculated_words_count = 0
word_index = 0
senti_dict = defaultdict(list)
for word in comment_parsed:
    word_POS = words_POS[word_index][1]
    # change POS to wordnet POS
    #POS_LIST =  nouns, verbs, adjectives and adverbs = n,v,a,r
    wn_word_POS = get_wordnet_pos(word_POS)
    #get synset of chosen POS
    wordnet = swn.senti_synsets(word, wn_word_POS)
    if len(list(wordnet)) > 0:
        wordnet = swn.senti_synsets(word, wn_word_POS)
        wordnet0 = list(wordnet)[0]
        print('positive proba for word {} is: {} '.format(word,wordnet0.pos_score()))
        senti_dict['pos'].append(wordnet0.pos_score())
        print('negative proba for word {} is: {} '.format(word, wordnet0.neg_score()))
        senti_dict['neg'].append(wordnet0.neg_score())
        print('objective proba for word {} is: {} '.format(word,wordnet0.obj_score()))
        senti_dict['obj'].append(wordnet0.obj_score())
        calculated_words_count += 1
    total_words_count += 1
    word_index+=1

print('total words in comment : {}'.format(total_words_count))
print('calculated words: {}, meaning {} of the post words'.format(calculated_words_count,"{:.1%}".format(calculated_words_count/total_words_count)))



