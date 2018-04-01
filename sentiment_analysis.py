import urllib.parse
import urllib.request
import pandas as pd
import nltk as nk
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def get_cosine(vec1, vec2):
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1])
    sum2 = sum([vec2[x] ** 2 for x in vec2])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


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


text_list = ["""b'I think most people would say that whether they act on desires and thoughts or not is precisely the point, for this and many other issues. Consider if your logic were used for other things:\n>I think everyone on a diet has desires/thoughts to eat unhealthy food. To me, whether they act on them or not is beside the point. It can really put me off eating healthy as I believe I and everyone else wants to eat fatty and unhealthy junk food.\n\nOr this:\n>I believe private property is false. I think everyone who is a law abiding citizen has desires/thoughts of stealing their neighbors\' cool stuff. To me, whether they act on them or not is beside the point. It can really put me off to buy nice things for myself as I believe everyone will be secretly jealous and want to steal my stuff. \n\nMany marriages are passionless and full of strife. However, this does not mean that open relationships will be full of passion and free of strife. \n\nRelationships are hard for many reasons. One of those reasons is that people usually get jealous or heartbroken when their partner forms a close connection with someone else. Monogamy is one way of addressing this difficulty: both partners make a commitment not to cheat during fleeting moments of passion in the hope that a long-term relationship will be more rewarding for both partners in the long run. Open relationships are another way of addressing this difficulty: both partners agree not to get their feelings hurt if their relationship doesn\'t last forever or if their relationship is shared with other people. Both of these "solutions" ask people to do something that is against human nature (not acting out on feelings and desires or not getting their feelings hurt).\n\nSo maybe you\'ll decide that monogamy is "false" (or at least, not right for you). But don\'t take that to automatically mean that polygamy or "swinging" or open relationships are "true" (or right for you). Many couples in "open" relationships still implicitly affirm at least some value in monogamous relationships by maintaining one committed, long-term relationship with each other while still agreeing that it is okay to flirt and have sex with other people so long as their primary partner consents""",
        """b"1. Long term economic disincentives will not be very effective because people are naturally optimistic, and will assume that if they have kids, their economic situation will improve even if they aren't currently able to financially support the child.\n\n2. Single people have more flexibility to find work, both in the hours of the day and the location in the country they can work.\n\n3. Punishing children for the mistakes of their parents is both unfair and economically unwise.  Children raised in warehouse daycares or neglectful households will do poorer at school and have less productive jobs overall.\n\n4. People with children are more likely to spend the tax break and stimulate the economy.""",
        """b'Excellent points. I would only add that most governments are going to try to add incentive to create families as it ensures future taxpayers. This is less of an argument and more of an explanation as to why families tend to get tax breaks, governments like them.'""",
        """b"I think there's only a very small chance that any of the 35 subscribers to this subreddit have any experience with your hometown. I feel like I could be relatively happy in any town with a movie theater, a few good restaurants, and some nice company. Is Oxford Hills lacking even these small requirements?""",
        """b'I feel as though the idea that without a "Leviathan" as you so eloquently put it, human beings would commit barbarous acts is a cultural myth similar to the boogey man. To me universal law doesn\'t need to be enforced by higher caste in order to be efficient. Most anarchist literature posits and idea of "protection agencies" or "factions" to enforce natural laws based around a central constitution where opression in all its forms is rejected. And the laws within are just implicitly understood."""]

sentiment_1 = np.array(sentiment_analysis(text_list[0])).reshape(1, -1)
print(sentiment_1)
sentiment_2 = np.array(sentiment_analysis(text_list[1])).reshape(1, -1)
print(sentiment_2)
sim = cosine_similarity(sentiment_1, sentiment_2)
print(sim)
