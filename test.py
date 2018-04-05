# import numpy as np
# import pandas as pd
# from collections import defaultdict
# import pickle
# import re
# # common_support_check = False
# import csv
import os
# import CreateFeatures
# from sklearn.feature_extraction.text import TfidfVectorizer
# import csv
# from optparse import OptionParser
# import sys
# from sklearn.metrics.pairwise import cosine_similarity
#
# #############################################test matching################################
# # # test trim common support
# # if common_support_check:
# #     d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
# #     data = pd.DataFrame(data = d)
# #
# #     group_min_max = (data.groupby('treatment')
# #                      .propensity.agg({"min_propensity": np.min, "max_propensity": np.max}))
# #
# #     min_common_support = np.max(group_min_max.min_propensity)
# #     print('min common support:{}'.format(min_common_support))
# #     max_common_support = np.min(group_min_max.max_propensity)
# #     print('max common support:{}'.format(max_common_support))
# #
# #     common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
# #     control = (data['treatment'] == 0)
# #     print('control:{}'.format(control))
# #     treated = (data['treatment'] == 1)
# #     print('treated:{}'.format(treated))
# #     print(data[common_support])
# # ######################################################################
# # d = {'propensity':[0.02,0.1,0.13,0.12,0.5,0.52,0.112,0.09,0.51,0.018],'treatment':[1,1,1,0,1,0,0,1,0,1]}
# # data = pd.DataFrame(data = d)
# #
# # # print((data.groupby("treatment")
# # #                          ["propensity"].agg({"min_propensity": np.min, "max_propensity": np.max})))
# # min_common_support = 0.2
# # print(data["propensity"] >= min_common_support)
# # #     data.propensity)
# # # print(data["propensity"])
#
# ########################################################################
#
# ###################################validate deltas##########################
# test_deltas = False
#
# if test_deltas:
#     comments = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                               "Causal-Inference-Final-Project\\"
#                                               "importing_change_my_view\\all submissions comments with label.csv", index_col=False)
#
#     pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
#     OP_deltas_comments_ids = pickle.load(pkl_file)
#
#     pkl_file = open('OP_deltas_comments_ids_deltalog.pickle', 'rb')
#     OP_deltas_comments_ids_deltalog = pickle.load(pkl_file)
#
#     # see label distribution in data
#     print("VALUE COUNT:")
#     print(comments['delta'].value_counts())
#
#     # validate that total number of deltas equal to the label distribution above
#     deltalog_manual_values_comment_list = list()
#     for key, value in OP_deltas_comments_ids_deltalog.items():
#         deltalog_manual_values_comment_list += value
#     deltalog_manual_values_comment_list = set(deltalog_manual_values_comment_list)
#     deltalog_manual_values_comment_list = list(deltalog_manual_values_comment_list)
#     print("num of deltas in deltalog: {}".format(len(deltalog_manual_values_comment_list)))
#
#     manual_deltas = list()
#     for k, value in OP_deltas_comments_ids.items():
#         value_stripped = [w.lstrip("b") for w in value]
#         value_stripped = [w.lstrip("'") for w in value]
#         manual_deltas += value_stripped
#     manual_deltas = set(manual_deltas)
#     manual_deltas = list(manual_deltas)
#     print("num of deltas in manual deltas: {}".format(len(manual_deltas)))
#
#     deltalog_manual_values_comment_list += manual_deltas
#     deltalog_manual_values_comment_list = list(set(deltalog_manual_values_comment_list))
#     print("TOTAL {} deltas".format(len(deltalog_manual_values_comment_list)))
#
#     #TODO: debug 2690 deltas in deltalog and not in data
#     deltalog_manual_values_comment_list = [w.lstrip("b") for w in deltalog_manual_values_comment_list]
#     deltalog_manual_values_comment_list = [w.strip("'") for w in deltalog_manual_values_comment_list]
#     deltalog_manual_values_comment_list = [w.lstrip("t1_") for w in deltalog_manual_values_comment_list]
#     manual_deltas = [w.lstrip("b") for w in manual_deltas]
#     manual_deltas = [w.strip("'") for w in manual_deltas]
#     manual_deltas = [w.lstrip("t1_") for w in manual_deltas]
#     manual_deltas = set(manual_deltas)
#     manual_deltas = list(manual_deltas)
#     deltalog_manual_values_comment_list = set(deltalog_manual_values_comment_list)
#     deltalog_manual_values_comment_list = list(deltalog_manual_values_comment_list)
#     deltas_from_labeled_data = comments[comments['delta'] == 1]["comment_id"]
#     deltas_from_labeled_data = [w.lstrip("b") for w in deltas_from_labeled_data]
#     deltas_from_labeled_data = [w.strip("'") for w in deltas_from_labeled_data]
#     deltas_from_labeled_data = [w.lstrip("t1_") for w in deltas_from_labeled_data]
#     both_lists = list(set(deltalog_manual_values_comment_list) | set(manual_deltas))
#     not_found = list(set(both_lists) - set(deltas_from_labeled_data))
#     print(comments[comments['comment_id'].str.contains('dpo25t0')]["comment_id"])
#
# ##############################################################################
#
# ##########################similarity feature######################################
#
# test_similarity = True
#
#
# if test_similarity:
#
#     base_directory = os.path.abspath(os.curdir)
#     results_directory = os.path.join(base_directory, 'importing_change_my_view')
#     comments = pd.read_csv(os.path.join(results_directory, 'all submissions comments with label.csv'))
#     submissions = pd.read_csv(os.path.join(results_directory, 'all submissions.csv'))
#
#     comments['submission_id'] = comments.submission_id.str.slice(2, -1)
#     submissions_drop = submissions.drop_duplicates(subset='submission_id', keep="last")
#
#     join_result = comments.merge(submissions_drop, on='submission_id', how='inner')
#
#     join_result["submmiter_commenter_tfidf_cos_sim"] = 0
#
#
#     def concat_df_rows(comment_created_utc, author, is_submission=False):
#         if is_submission:
#             text = join_result.loc[(join_result['comment_created_utc'] < comment_created_utc) &
#                                    (join_result['comment_author'] == author)].iloc[0][["submission_title",
#                                                                                        "submission_body"]]
#             text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
#             text["submission_body"] = text["submission_body"].partition(
#                 "Hello, users of CMV! This is a footnote from your moderators")[0]
#             text["submission_title_and_body"] = text["submission_title"] + text["submission_body"]
#             text_cat = text.str.cat(sep=' ')
#             return text_cat
#
#         text = join_result.loc[(join_result['comment_created_utc'] < comment_created_utc) &
#                                (join_result['comment_author'] == author)]["comment_body"]
#         text = text.apply(lambda x: x.lstrip('b').strip('"').strip("'").strip(">"))
#         text_cat = text.str.cat(sep=' ')
#
#         return text_cat
#
#
#     for index, row in join_result.iterrows():
#         # define thresholds anf filter variables
#         comment_created_utc = row.loc['comment_created_utc']
#         comment_author = row.loc['comment_author']
#         submission_author =  row.loc['submission_author']
#
#         # all text of commenter until comment time
#         text_commenter = concat_df_rows(comment_created_utc, comment_author)
#
#         # all text of submissioner until comment time
#         text_submissioner = concat_df_rows(comment_created_utc, submission_author)
#         text_submissioner_submission = concat_df_rows(comment_created_utc, submission_author, True)
#         text_submissioner += text_submissioner_submission
#
#         text = [text_submissioner, text_commenter]
#         tfidf = TfidfVectorizer(stop_words = 'english', lowercase = True, analyzer  = 'word', norm = 'l2',
#                                 smooth_idf = True, sublinear_tf  = False, use_idf  = True).fit_transform(text)
#         similarity = cosine_similarity(tfidf[0:], tfidf[1:])
#
#         join_result["submmiter_commenter_tfidf_cos_sim"].loc[index, "submmiter_commenter_tfidf_cos_sim"] = \
#             similarity[0][0]
#
#         if similarity[0][0] > 0.9 or similarity[0][0] < 0.2:
#             print(similarity[0][0])
#             print("text submissioner:")
#             print(text_submissioner)
#             print("text commenter:")
#             print(text_commenter)
#
# ##################################################################################################


# import pandas as pd
# import os
#
# base_directory = os.path.abspath(os.curdir)
# change_my_view_directory = os.path.join(base_directory, 'change my view')
# data = pd.read_csv(os.path.join(change_my_view_directory, 'data_label_treatment_all.csv'))
#
# treatment = data.loc[data['treated'] == 1]
# print('size of treated = {}'.format(treatment.shape))
# control = data.loc[data['treated'] == 0]
# print('size of control = {}'.format(control.shape))
# no_parent = data.loc[data['treated'] == -1]
# print('size of no_parent = {}'.format(no_parent.shape))
#
# treatment_delta = data.loc[(data['treated'] == 1) & (data['delta'] == 1)]
# print('size of treated_delta = {}'.format(treatment_delta.shape))
# treatment_no_delta = data.loc[(data['treated'] == 1) & (data['delta'] == 0)]
# print('size of treatment_no_delta = {}'.format(treatment_no_delta.shape))
# control_delta = data.loc[(data['treated'] == 0) & (data['delta'] == 1)]
# print('size of control_delta = {}'.format(control_delta.shape))
# control_no_delta = data.loc[(data['treated'] == 0) & (data['delta'] == 0)]
# print('size of control_no_delta = {}'.format(control_no_delta.shape))

# import datetime
# import time
# import pytz
# import math
#
#
# tz = pytz.timezone('GMT')  # America/New_York
# date_comment = datetime.datetime.fromtimestamp(1521029983, tz)
# utc_now = int(time.time())
# date_now = datetime.datetime.fromtimestamp(utc_now, tz)
# c = date_now - date_comment
# months = divmod(c.days * 86400 + c.seconds, 60)
# print(c.days/31)

import urllib.parse
import urllib.request
import pandas as pd
import nltk as nk
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.sklearn_api import ldamodel
from gensim.models import LdaModel

# ps = PorterStemmer()
#
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
# hotel_rev = ['football is vary great sport, it has balls and sport for everyone',
#              'sport is so magnificent',
#              'sport is great, we eat a lot of sugar and food and resturant',
#              'sugar is fun, we eat there a lot of food and sugar and sport',
#              'sugar is sweet, I love eat it, it is my favorite food and bread and resturant',
#              'sugar is biter, I love eat it, it is my favorite food and resturant']
#
# sid = SentimentIntensityAnalyzer()
# for sentence in hotel_rev:
#     print(sentence)
#     result = sid.polarity_scores(sentence)
#     neg_prob = result['neg']
#     neutral_prob = result['neu']
#     pos_prob = result['pos']
#     final = [pos_prob, neg_prob, neutral_prob]
#     for k in result:
#         print('{0}: {1}, '.format(k, result[k]), end='')
#         print()


# def sentiment_analysis(text):
#     data = urllib.parse.urlencode({"text": text})
#     data = data.replace("\n", "")
#     data = data.lower()
#     data = data.encode('ascii')
#     with urllib.request.urlopen("http://text-processing.com/api/sentiment/", data) as f:
#         result = f.read().decode('utf-8')
#         index_of_label = result.find('label')
#         label = result[index_of_label + 9:index_of_label + 12]
#         label_list = semantic_result(label)
#         return label_list
#
#
# def semantic_result(semantic):
#     return {
#         'pos': [1, 0, 0],
#         'neg': [0, 1, 0],
#         'neu': [0, 0, 1]
#     }[semantic]
#
#
# def parse_comment(comment):
#     # get POS
#     comment_parsed = nk.word_tokenize(comment)
#     # print('parsed comment is: {}'.format(comment_parsed))
#
#     return comment_parsed
#
#
# def get_POS(comment_parsed):
#     words_pos = nk.pos_tag(comment_parsed)
#
#     return words_pos
#
#
# table = [['Heading1', 'Heading2'], [1, 2], [3, 4]]
# headers = table.pop(0)
# df = pd.DataFrame(table, columns=headers)
# df.assign(reut='', shimon='')

#
# for text in text_list:
#     parsed_text = parse_comment(text)
#     pos_text = get_POS(parsed_text)
#     pos_df = pd.DataFrame(pos_text, columns=['word', 'POS'])
#     all_tags = pos_df.shape[0]
#     all_pos = pos_df['POS']
#     start_time = time.time()
#     adj = pos_df.loc[pos_df.POS.isin(['JJ', 'JJS', 'JJR'])]
#     adj_pos = adj.shape[0]
#     print("My program took", time.time() - start_time, "to run")
#     print(adj_pos)
#
#     start_time2 = time.time()
#     freq = nk.FreqDist(all_pos)
#     adj_pos2 = freq['JJ'] + freq['JJS'] + freq['JJR']
#     print("My program took", time.time() - start_time2, "to run")
#     print(adj_pos2)

# for index, row in df.iterrows():
#     label_res = sentiment_analysis(text)
#     row['reut'], row['shimon'] = label_res[0], label_res[1]
#     reut = 1


# text = """b'I think most people would say that whether they act on desires and thoughts or not is precisely the point, for this and many other issues. Consider if your logic were used for other things:\n>I think everyone on a diet has desires/thoughts to eat unhealthy food. To me, whether they act on them or not is beside the point. It can really put me off eating healthy as I believe I and everyone else wants to eat fatty and unhealthy junk food.\n\nOr this:\n>I believe private property is false. I think everyone who is a law abiding citizen has desires/thoughts of stealing their neighbors\' cool stuff. To me, whether they act on them or not is beside the point. It can really put me off to buy nice things for myself as I believe everyone will be secretly jealous and want to steal my stuff. \n\nMany marriages are passionless and full of strife. However, this does not mean that open relationships will be full of passion and free of strife. \n\nRelationships are hard for many reasons. One of those reasons is that people usually get jealous or heartbroken when their partner forms a close connection with someone else. Monogamy is one way of addressing this difficulty: both partners make a commitment not to cheat during fleeting moments of passion in the hope that a long-term relationship will be more rewarding for both partners in the long run. Open relationships are another way of addressing this difficulty: both partners agree not to get their feelings hurt if their relationship doesn\'t last forever or if their relationship is shared with other people. Both of these "solutions" ask people to do something that is against human nature (not acting out on feelings and desires or not getting their feelings hurt).\n\nSo maybe you\'ll decide that monogamy is "false" (or at least, not right for you). But don\'t take that to automatically mean that polygamy or "swinging" or open relationships are "true" (or right for you). Many couples in "open" relationships still implicitly affirm at least some value in monogamous relationships by maintaining one committed, long-term relationship with each other while still agreeing that it is okay to flirt and have sex with other people so long as their primary partner consents"""
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
#     neg_prob = float(result[index_of_neg + 6:index_of_neutral-3])
#     print(neg_prob, type(neg_prob))
#     neutral_prob = float(result[index_of_neutral + 10:index_of_pos-3])
#     print(neutral_prob, type(neutral_prob))
#     pos_prob = float(result[index_of_pos + 6:index_of_label-4])
#     print(pos_prob, type(pos_prob))
#     print(pos_prob - neutral_prob)


# def get_cosine(vec1, vec2):
#     intersection = set(vec1) & set(vec2)
#     numerator = sum([vec1[x] * vec2[x] for x in intersection])
#
#     sum1 = sum([vec1[x] ** 2 for x in vec1])
#     sum2 = sum([vec2[x] ** 2 for x in vec2])
#     denominator = math.sqrt(sum1) * math.sqrt(sum2)
#
#     if not denominator:
#         return 0.0
#     else:
#         return float(numerator) / denominator
#
#
# def sentiment_analysis(text):
#     data = urllib.parse.urlencode({"text": text})
#     data = data.replace("\n", "")
#     data = data.lower()
#     data = data.encode('ascii')
#     with urllib.request.urlopen("http://text-processing.com/api/sentiment/", data) as f:
#         result = f.read().decode('utf-8')
#         index_of_neg = result.find('neg')
#         index_of_neutral = result.find('neutral')
#         index_of_pos = result.find('pos')
#         index_of_label = result.find('label')
#         neg_prob = float(result[index_of_neg + 6:index_of_neutral - 3])
#         neutral_prob = float(result[index_of_neutral + 10:index_of_pos - 3])
#         pos_prob = float(result[index_of_pos + 6:index_of_label - 4])
#         return [pos_prob, neg_prob, neutral_prob]
#
#
# # text_list = ["""b'I think most people would say that whether they act on desires and thoughts or not is precisely the point, for this and many other issues. Consider if your logic were used for other things:\n>I think everyone on a diet has desires/thoughts to eat unhealthy food. To me, whether they act on them or not is beside the point. It can really put me off eating healthy as I believe I and everyone else wants to eat fatty and unhealthy junk food.\n\nOr this:\n>I believe private property is false. I think everyone who is a law abiding citizen has desires/thoughts of stealing their neighbors\' cool stuff. To me, whether they act on them or not is beside the point. It can really put me off to buy nice things for myself as I believe everyone will be secretly jealous and want to steal my stuff. \n\nMany marriages are passionless and full of strife. However, this does not mean that open relationships will be full of passion and free of strife. \n\nRelationships are hard for many reasons. One of those reasons is that people usually get jealous or heartbroken when their partner forms a close connection with someone else. Monogamy is one way of addressing this difficulty: both partners make a commitment not to cheat during fleeting moments of passion in the hope that a long-term relationship will be more rewarding for both partners in the long run. Open relationships are another way of addressing this difficulty: both partners agree not to get their feelings hurt if their relationship doesn\'t last forever or if their relationship is shared with other people. Both of these "solutions" ask people to do something that is against human nature (not acting out on feelings and desires or not getting their feelings hurt).\n\nSo maybe you\'ll decide that monogamy is "false" (or at least, not right for you). But don\'t take that to automatically mean that polygamy or "swinging" or open relationships are "true" (or right for you). Many couples in "open" relationships still implicitly affirm at least some value in monogamous relationships by maintaining one committed, long-term relationship with each other while still agreeing that it is okay to flirt and have sex with other people so long as their primary partner consents""",
# #         """b"1. Long term economic disincentives will not be very effective because people are naturally optimistic, and will assume that if they have kids, their economic situation will improve even if they aren't currently able to financially support the child.\n\n2. Single people have more flexibility to find work, both in the hours of the day and the location in the country they can work.\n\n3. Punishing children for the mistakes of their parents is both unfair and economically unwise.  Children raised in warehouse daycares or neglectful households will do poorer at school and have less productive jobs overall.\n\n4. People with children are more likely to spend the tax break and stimulate the economy.""",
# #         """b'Excellent points. I would only add that most governments are going to try to add incentive to create families as it ensures future taxpayers. This is less of an argument and more of an explanation as to why families tend to get tax breaks, governments like them.'""",
# #         """b"I think there's only a very small chance that any of the 35 subscribers to this subreddit have any experience with your hometown. I feel like I could be relatively happy in any town with a movie theater, a few good restaurants, and some nice company. Is Oxford Hills lacking even these small requirements?""",
# #         """b'I feel as though the idea that without a "Leviathan" as you so eloquently put it, human beings would commit barbarous acts is a cultural myth similar to the boogey man. To me universal law doesn\'t need to be enforced by higher caste in order to be efficient. Most anarchist literature posits and idea of "protection agencies" or "factions" to enforce natural laws based around a central constitution where opression in all its forms is rejected. And the laws within are just implicitly understood."""]
#
#
#
# # sentiment_1 = np.array(sentiment_analysis(text_list[0])).reshape(1, -1)
# # print(sentiment_1)
# # sentiment_2 = np.array(sentiment_analysis(text_list[1])).reshape(1, -1)
# # print(sentiment_2)
# # sim = cosine_similarity(sentiment_1, sentiment_2)
# # print(sim)
#
# stop = set(stopwords.words('english'))
# exclude = set(string.punctuation)
# lemma = WordNetLemmatizer()
#
#
# def clean(doc):
#     text = doc.lstrip('b').strip('"').strip("'").strip(">")
#     stop_free = " ".join([i for i in text.lower().split() if i not in stop])
#     punc_free = "".join(ch for ch in stop_free if ch not in exclude)
#     # stemming = " ".join(ps.stem(word) for word in punc_free.split())
#     normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
#     return normalized
#
#
# text_list = ['football is a great sport, it has balls and sport for everyone',
#              'sport is great, there are sport and football',
#              'sport is great, we eat a lot of sugar and food and resturant',
#              'sugar is fun, we eat there a lot of food and sugar and sport',
#              'sugar is sweet, I love eat it, it is my favorite food and bread and resturant',
#              'sugar is biter, I love eat it, it is my favorite food and resturant']
#
# import os
# base_directory = os.path.abspath(os.curdir)
# data_directory = os.path.join(base_directory, "change my view")
#
# units = pd.read_excel(os.path.join(data_directory, 'small_data.xlsx'))
#
# # doc_clean = {index: clean(doc).split() for index, doc in enumerate(text_list)}
# comments_clean = {row['comment_id']: clean(row['comment_body']).split() for index, row in units.iterrows()}
# # Creating the term dictionary of our corpus, where every unique term is assigned an index.
# dictionary = corpora.Dictionary(comments_clean.values())
#
# # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# comment_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in comments_clean.items()}
#
# # Creating the object for LDA model using gensim library
# Lda = gensim.models.ldamodel.LdaModel
# # model = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50, eta=0.1)
# model = ldamodel.LdaTransformer(num_topics=3, id2word=dictionary, passes=50, minimum_probability=0)
# model = model.fit(comment_term_matrix.values())
# # Running and Trainign LDA model on the document term matrix.
# result = model.transform(list(comment_term_matrix.values()))
# print(result)
#
# # result = model[doc_term_matrix[0]]
# # print(result)
#
# #
#
# import pandas as pd
#
# check = pd.DataFrame({'one': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),
#                       'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])})
#
# df_empty = pd.DataFrame()
# start_time = time.time()
# print(int(check.empty))
# # print(int(df_empty.empty))
# empty_time = time.time()
# print('time for empty: ', empty_time - start_time)
# if check.empty:
#     print(1)
# else:
#     print(0)
#
# empty_time1 = time.time()
# print('time for empty: ', empty_time1 - empty_time)


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'change my view')
check = pd.read_csv(os.path.join(data_directory, 'final_df_CMV_after_fix.csv'), encoding='utf-8')
print(check.min())

reut = 1