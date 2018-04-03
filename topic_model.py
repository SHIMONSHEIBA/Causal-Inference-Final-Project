from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
from gensim.sklearn_api import ldamodel
import os
import pandas as pd
import time


base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, "change my view")


class TopicModel:
    def __init__(self, number_of_topics):
        self.units = pd.read_csv(os.path.join(data_directory, 'units_0304.csv'))
        self.all_data = pd.read_csv(os.path.join(data_directory, 'all_data_0304.csv'))
        self.stop = set(stopwords.words('english'))
        self.exclude = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        self.number_of_topics = number_of_topics

    def clean(self, doc):
        text = doc.lstrip('b').strip('"').strip("'").strip(">")
        stop_free = " ".join([i for i in text.lower().split() if i not in self.stop])
        punc_free = "".join(ch for ch in stop_free if ch not in self.exclude)
        # stemming = " ".join(ps.stem(word) for word in punc_free.split())
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    def topic_model(self):
        # Clean the data
        print('{}: Clean the data'.format((time.asctime(time.localtime(time.time())))))
        units_clean = {row['comment_id']: self.clean(row['comment_body']).split()
                       for index, row in self.units.iterrows()}
        all_data_clean = {row['comment_id']: self.clean(row['comment_body']).split()
                          for index, row in self.all_data.iterrows()}
        # Creating the term dictionary of our corpus, where every unique term is assigned an index.
        print('{}: Create the dictionary'.format((time.asctime(time.localtime(time.time())))))
        dictionary = corpora.Dictionary(all_data_clean.values())

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        print('{}: Create units term matrix'.format((time.asctime(time.localtime(time.time())))))
        units_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in units_clean.items()}
        print('{}: Create all data term matrix'.format((time.asctime(time.localtime(time.time())))))
        all_data_term_matrix = {index: dictionary.doc2bow(doc) for index, doc in all_data_clean.items()}

        # Create LDA model
        print('{}: Create model'.format((time.asctime(time.localtime(time.time())))))
        model = ldamodel.LdaTransformer(num_topics=self.number_of_topics, id2word=dictionary, passes=50,
                                        minimum_probability=0)
        # Train LDA model on the comments term matrix.
        print('{}: Fit the model on all data'.format((time.asctime(time.localtime(time.time())))))
        model = model.fit(list(all_data_term_matrix.values()))
        # Get topics for the data
        print('{}: Predict topics for units'.format((time.asctime(time.localtime(time.time())))))
        result = model.transform(list(units_term_matrix.values()))

        print('{}: Create final topic model data'.format((time.asctime(time.localtime(time.time())))))
        comment_ids_df = pd.DataFrame(list(units_term_matrix.keys()), columns=['comment_id'])
        result_columns = ['topic_model_'+str(i) for i in range(self.number_of_topics)]
        topic_model_result_df = pd.DataFrame(result, columns=result_columns)

        print('{}: Save final topic model data'.format((time.asctime(time.localtime(time.time())))))
        topic_model_final_result = pd.concat([comment_ids_df, topic_model_result_df], axis=1)
        topic_model_final_result.to_csv(os.path.join(data_directory, 'topic_model_CMV.csv'))


def main():
    number_of_topics = 10
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    topic_model_obj = TopicModel(number_of_topics)
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: all data: {}, units data: {}'.format(topic_model_obj.all_data.shape,
                                                            topic_model_obj.units.shape))
    topic_model_obj.topic_model()


if __name__ == '__main__':
    main()
