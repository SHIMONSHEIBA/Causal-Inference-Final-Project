
#first run this, download() will open an installation gui
# import nltk
# nltk.download()

# after installation you will be able to import this:
from nltk.corpus import sentiwordnet as swn


comment = 'sub as well and think it is fun to check out peoples instagram accounts for new ideas; for example, look at the beautiful and low calorie dishes on this persons feed'
comment_parsed = comment.split(' ')
print(comment_parsed)

total_words_count = 0
calculated_words_count = 0
for word in comment_parsed:
    wordnet = swn.senti_synsets(word, 'a')
    if len(list(wordnet)) > 0:
        wordnet = swn.senti_synsets(word, 'a')
        wordnet0 = list(wordnet)[0]
        print('positive proba for word {} is: {} '.format(word,wordnet0.pos_score()))
        print('negative proba for word {} is: {} '.format(word, wordnet0.neg_score()))
        print('objective proba for word {} is: {} '.format(word,wordnet0.obj_score()))
        calculated_words_count += 1
    total_words_count += 1

print('total words in comment : {}'.format(total_words_count))
print('calculated words: {}, meaning {}'.format(calculated_words_count,"{:.1%}".format(calculated_words_count/total_words_count)))


# all = swn.all_senti_synsets()

