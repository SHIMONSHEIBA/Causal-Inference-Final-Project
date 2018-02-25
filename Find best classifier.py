"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import scipy


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--pct_features",
              action="store", type="int", dest="pct_features", default=0.75,
              help="Select % number of features using a feature selection")
op.add_option("--chi2_use",
              action="store", dest="is_chi2",
              help="Whether to use a chi-squared test for feature selection")
op.add_option("--SVD_use",
              action="store", dest="is_SVD",
              help="Whether to use SVD for feature selection")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
op.add_option("--k_fold",
              action='store', type=int, default=100,
              help='k_fold when using cross validation')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


###############################################################################
# Load some categories from the training set
# if opts.all_categories:
#     categories = None
# else:
#     categories = [
#         'alt.atheism',
#         'talk.religion.misc',
#         'comp.graphics',
#         'sci.space',
#     ]

# if opts.filtered:
#     remove = ('headers', 'footers', 'quotes')
# else:
#     remove = ()


###############################################################################
class Classifier:
    def __init__(self):
        self.X_train = None
        self.feature_names = None
        print('loading the data')
        if True:  # subject == 'diet':
            with open('old data/diet_keto_percent_before_after_reference.csv', 'r') as csvfile:
                dataMatrix = list(csv.reader(csvfile))

            i = 0
            for comment in dataMatrix:  # delete the header of the file
                if comment[0] == 'comment_body':
                    del dataMatrix[i]
                    break
                i += 1
            print('create numpy types')
            dataMatrix = np.array(dataMatrix)
            self.data_train = dataMatrix[:, 0]
            self.target_names = dataMatrix[:, 1]
            self.pct_before = dataMatrix[:, 2].astype(float)
            self.pct_after = dataMatrix[:, 4].astype(float)
            self.comment_lenght = dataMatrix[:, 3].astype(int)
            self.target_names = self.target_names.astype(int)
            csvfile.close()
        print('data loaded')
        return


    def featureSelection(self, use_hashing, use_chi2):
        print("Extracting features from the training data using a sparse vectorizer")
        # t0 = time()
        opts.use_hashing = use_hashing
        if opts.use_hashing:
            vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                           n_features=opts.n_features, ngram_range=(1, 3))
            self.X_train = vectorizer.transform(self.data_train)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english', ngram_range=(1, 3))
            self.X_train = vectorizer.fit_transform(self.data_train)

        print("n_samples: %d, n_features: %d" % self.X_train.shape)
        print()

        # mapping from integer feature name to original token string
        if opts.use_hashing:
            self.feature_names = None
        else:
            self.feature_names = vectorizer.get_feature_names()

        if use_chi2:
            opts.is_chi2 = True
            opts.is_SVD = False
        else:
            opts.is_chi2 = False
            opts.is_SVD = True

        if opts.is_chi2:
            print("Extracting %d best features by a chi-squared test" %
                  int(opts.pct_features * self.X_train.shape[1]))
            t0 = time()
            ch2 = SelectKBest(chi2, k=int(opts.pct_features*self.X_train.shape[1]))
            self.X_train = ch2.fit_transform(self.X_train, self.target_names)
            if self.feature_names:
                # keep selected feature names
                self.feature_names = [self.feature_names[i] for i
                                 in ch2.get_support(indices=True)]
            print("done in %fs" % (time() - t0))
            print()

        if opts.is_SVD:
            print("Extracting %f best features by SVD" %
                  opts.pct_features)
            t0 = time()
            print('number of feature to select is: %d' % int(opts.pct_features*self.X_train.shape[1]))
            svd = TruncatedSVD(n_components=int(opts.pct_features*self.X_train.shape[1]), n_iter=7)
            self.X_train = svd.fit_transform(self.X_train, self.target_names)

            print("done in %fs" % (time() - t0))
            print()

            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

            print()

        if self.feature_names:
            self.feature_names = np.asarray(self.feature_names)

        self.pct_before = scipy.sparse.csr_matrix(self.pct_before)
        self.pct_after = scipy.sparse.csr_matrix(self.pct_after)
        self.X_train = scipy.sparse.hstack((self.X_train, self.pct_before.T, self.pct_after.T), format='csr')
        self.X_train = scipy.sparse.hstack((self.X_train, self.pct_before.T), format='csr')


###############################################################################
# benchmark classifiers
    def benchmark(self, clf, clf_name='default'):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        # Cross validation part
        k = 10
        # print('cross validation with %d folds' % k)
        predicted = range(self.X_train.shape[0])
        probability = range(self.X_train.shape[0])
        prob_list = range(self.X_train.shape[0])
        if clf_name == 'GaussianNB':
            self.X_train = self.X_train.toarray()
        # predicted = cross_val_predict(clf, self.X_train, self.target_names, cv=k)
        # if clf_name != 'MultinomialNB':
        #     print("Don't train")
        #     return clf_name, 0, 0, 0
        for train_index, test_index in LeaveOneOut().split(self.X_train):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.target_names[train_index], self.target_names[test_index]
            clf.fit(X_train, y_train)
            predicted[test_index] = clf.predict(X_test)
            if clf_name == 'MultinomialNB':
                probability[test_index] = clf.predict_proba(X_test)
            # auc.append(metrics.roc_auc_score(y_test, predicted, average='samples'))
        score = metrics.accuracy_score(self.target_names, predicted)
        train_time = time() - t0
        print("cross validation time: %0.3fs" % train_time)
        # if hasattr(clf, 'coef_'):
        #     print("dimensionality: %d" % clf.coef_.shape[1])
        #     print("density: %f" % density(clf.coef_))
        #
        #     if opts.print_top10 and self.feature_names is not None:
        #         print("top 10 keywords per class:")
        #         for i, label in enumerate(self.target_names):
        #             top10 = np.argsort(clf.coef_[i])[-10:]
        #             print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
        #     print()

        # if True:  # opts.print_report:
        #     print("classification report:")
        #     print(metrics.classification_report(self.target_names, predicted,
        #                                             self.target_names=self.target_names))

        if opts.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.target_names, predicted, labels=[-1, 1]))

            print()
            clf_descr = str(clf).split('(')[0]
        print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

        auc = metrics.roc_auc_score(self.target_names, predicted, average='samples')
        print('AUC: %0.2f' % auc)

        if clf_name == 'MultinomialNB':
            for i in range(0, self.X_train.shape[0]):
                if predicted[i] == 1:
                    prob_list[i] = probability[i][0][1]
                else:
                    prob_list[i] = -1.0 * probability[i][0][0]
            false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(self.target_names, prob_list)
            roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
            plt.title('Receiver Operating Characteristic for Multinomial NB')
            plt.plot(false_positive_rate, true_positive_rate, 'b',
                     label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.2])
            plt.ylim([-0.1, 1.2])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            plt.savefig('ROC_for_MultinomialNB.png', bbox_inches='tight')

        # average_acc = 1.0*sum(score) / len(score)
        # print("Accuracy: %f" % average_acc)

        # average_auc = sum(auc) / len(auc)
        # print("AUC: %d" % average_auc)
        # auc = 1
        # clf_descr = str(clf).split('(')[0]

        return clf_descr, score, auc, train_time

    def ModelsIteration(self):
        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                # (RandomForestClassifier(n_estimators=100), "Random forest"),
                (SVC(C=1e-8, gamma=1.0/self.X_train.shape[1], kernel='rbf'), "SVM with rbf kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf, name))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            # results.append(self.benchmark(LinearSVC(loss='l2', penalty=penalty,
            #                                         dual=False, tol=1e-3), 'LinearSVC'))
            results.append(self.benchmark(LinearSVC(), 'LinearSVC'))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                   penalty=penalty), 'SGDClassifier'))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01), 'MultinomialNB'))
        results.append(self.benchmark(BernoulliNB(alpha=.01), 'BernoulliNB'))
        # results.append(self.benchmark(GaussianNB(), 'GaussianNB'))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(Pipeline([
          ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
          ('classification', LinearSVC())
        ])))

        # make some plots

        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]
        with open('pythonResultsCv15.csv', "wb") as ResultsFile:
            writer = csv.writer(ResultsFile)
            writer.writerows(results)
        ResultsFile.close()

        clf_names, score, auc, training_time = results
        training_time = np.array(training_time) / np.max(training_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        # plt.barh(indices + .3, training_time, .2, label="training time",
        #          color='c')
        plt.barh(indices + .3, auc, .2, label="ACU", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()
        plt.savefig('pythonResultsCv15.png', bbox_inches='tight')
        return


if __name__ == '__main__':
    classifier = Classifier()
    classifier.featureSelection(False, True)
    classifier.ModelsIteration()
