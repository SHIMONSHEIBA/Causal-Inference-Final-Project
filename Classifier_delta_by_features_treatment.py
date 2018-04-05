from optparse import OptionParser
import sys
from time import time
import time
from datetime import datetime
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
import itertools
import logging
import os
import math
from xgboost.sklearn import XGBClassifier

# Display progress logs on stdout
base_directory = os.path.abspath(os.curdir)
logs_directory = os.path.join(base_directory, 'logs')
results_directory = os.path.join(base_directory, 'choose_model_results')
features_directory = os.path.join(base_directory, 'change my view')
LOG_FILENAME = os.path.join(logs_directory, datetime.now().strftime('LogFile_causality_data_%d%m%Y%H%M.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,)

# parse commandline arguments
op = OptionParser()
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--k_fold",
              action='store', type=int, default=5,
              help='k_fold when using cross validation')
op.add_option("--split_Peff",
              action='store', default=True,
              help='whether to create classifier for each Peff group')
op.add_option("--is_backward",
              action='store', default=True,
              help='whether to use backward elimination or forward selection, if True - use backward elimination')
op.add_option("--is_backward_forward",
              action='store', default=False,
              help='whether to use backward or forward elimination selection, if False - dont use either')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()


###############################################################################
class Classifier:
    def __init__(self, label_column_name):
        self.X_train = None
        self.features = None
        self.feature_names = None
        print('{}: Loading the data: final_features_causality'.format((time.asctime(time.localtime(time.time())))))
        self.labels = None
        self.featuresDF = pd.read_csv(os.path.join(features_directory,
                                                   'matches_data_frame_treated_propensity_score_treated_logistic_all_deltas.csv'))
        self.label_column_name = label_column_name

        # group_dict is in the format: {index: [features list of this group], group name
        self.group_dict = {0: [['commenter_number_submission', 'commenter_number_comment',
                               'number_of_comments_in_tree_by_comment_user', 'commenter_seniority_days'],
                               'commenter_features'],
                           1: [['submitter_number_submission', 'submitter_seniority_days', 'submitter_number_comment',
                               'number_of_comments_in_tree_from_submitter', 'number_of_respond_by_submitter_total',
                                'number_of_respond_by_submitter'], 'submitter_features'],
                           2: [['is_first_comment_in_tree', 'comment_len',  'comment_depth'], 'comment_features'],
                           3: [['time_ratio', 'time_between_messages', 'time_until_first_comment',
                               'time_between_comment_first_comment'], 'time_features'],
                           4: [['submission_len', 'title_len'], 'submission_features'],
                           5: [['respond_to_comment_user_responses_ratio', 'respond_to_comment_user_all_ratio',
                               'respond_total_ratio'], 'ratio_features'],
                           6: [['treated'], 'trated'],
                           7: [['nltk_com_sen_pos', 'nltk_com_sen_neg', 'nltk_com_sen_neutral', 'nltk_sub_sen_pos',
                               'nltk_sub_sen_neg', 'nltk_sub_sen_neutral', 'nltk_title_sen_pos', 'nltk_title_sen_neg',
                                'nltk_title_sen_neutral', 'nltk_sim_sen'], 'sentiment features'],
                           8: [['percent_adj'], 'percent_adj'],
                           9: [['submmiter_commenter_tfidf_cos_sim'], 'submitted_commenter_similarity'],
                           10: [['topic_model_0', 'topic_model_1', 'topic_model_2', 'topic_model_3', 'topic_model_4',
                                'topic_model_5', 'topic_model_6', 'topic_model_7', 'topic_model_8', 'topic_model_9',
                                 'topic_model_10', 'topic_model_11', 'topic_model_12', 'topic_model_13',
                                 'topic_model_14'], 'topic_model']
                           }

        print('{}: Data loaded '.format((time.asctime(time.localtime(time.time())))))
        return

###############################################################################
    def split_relevant_data(self):
        """
        This function split the data to opts.k_fold folders and insert the group number to the DF
        :return:
        """
        # Split the data to k=opts.k_fold groups, each comment_author in one group only
        i = 1
        number_sample_group = 0
        sample_per_group = math.floor(self.featuresDF.shape[0] / opts.k_fold)
        self.featuresDF = self.featuresDF.sample(frac=1).reset_index(drop=True)
        for index, row in self.featuresDF.iterrows():
            if number_sample_group < sample_per_group or i == opts.k_fold:
                self.featuresDF.set_value(index, 'group_number', i)
                number_sample_group += 1
            else:
                i += 1
                self.featuresDF.set_value(index, 'group_number', i)
                print('{}: finish split samples for group number {} with {} samples'.
                      format((time.asctime(time.localtime(time.time()))), i-1, number_sample_group))
                print('{}: start split samples for group number {}'.
                      format((time.asctime(time.localtime(time.time()))), i))
                logging.info('{}: finish split samples for group number {} with {} samples'.
                             format((time.asctime(time.localtime(time.time()))), i-1, number_sample_group))
                logging.info('{}: start split samples for group number {}'.
                             format((time.asctime(time.localtime(time.time()))), i))
                number_sample_group = 1
        opts.k_fold = i + 1
        # print for the last group
        print('{}: finish split samples for group number {} with {} samples'.
              format((time.asctime(time.localtime(time.time()))), i, number_sample_group))
        logging.info('{}: finish split samples for group number {} with {} samples'.
                     format((time.asctime(time.localtime(time.time()))), i, number_sample_group))
        self.labels = self.featuresDF[[self.label_column_name, 'group_number']]
        print('{}: Finish split the data'.format((time.asctime(time.localtime(time.time())))))
        logging.info('{}: Finish split the data'.format((time.asctime(time.localtime(time.time())))))


###############################################################################
    def create_data_no_feature_selection(self):
        """
        This function create the data for the models if not using feature selection
        :return:
        """
        selected_features = list(self.group_dict.keys())
        features_group = [self.group_dict[group][0] for group in selected_features]
        self.features = [item for sublist in features_group for item in sublist]
        features = [item for sublist in features_group for item in sublist]
        features.append('group_number')
        self.X_train = self.featuresDF[features]
        features_names = [self.group_dict[feature][1] for feature in selected_features]
        print('{}: Start training with the groups: {}'.format((time.asctime(time.localtime(time.time()))),
                                                              features_names))
        logging.info('{}: Start training with the groups: {}'
                     .format((time.asctime(time.localtime(time.time()))), features_names))
        group_results = self.models_iteration()

        for model in group_results:
            model.append(features_names)
            model.append(opts.k_fold)
        columns_names = ['classifier_name', 'score', 'auc', 'train_time', 'features_list', 'k_fold']
        group_results_df = pd.DataFrame(group_results, columns=columns_names)

        return group_results_df

    def iterate_over_features_groups(self):
        """
        This function create the data for the models if using feature selection
        :return:
        """
        all_groups_results = pd.DataFrame()
        remaining_features = list(self.group_dict.keys())
        if opts.is_backward:  # use backward elimination or none of them
            selected_features = list(self.group_dict.keys())
        else:  # use forward selection
            selected_features = []
            remaining_features = [x for x in remaining_features if x not in selected_features]
        current_auc, best_new_auc = 0.0, 0.0
        remain_number_of_candidate = len(remaining_features)
        while remaining_features and current_auc == best_new_auc and remain_number_of_candidate > 0:
            auc_with_candidates = list()
            for candidate in remaining_features:
                if opts.is_backward:  # use backward elimination
                    features_group = [self.group_dict[group][0] for group in selected_features]
                    features_group.remove(self.group_dict[candidate][0])
                    self.features = [item for sublist in features_group for item in sublist]
                    features = [item for sublist in features_group for item in sublist]
                    features.append('group_number')
                    self.X_train = self.featuresDF[features]
                    features_names = [self.group_dict[feature][1] for feature in selected_features]
                    features_names.remove(self.group_dict[candidate][1])

                else:  # use forward selection
                    features_group = [self.group_dict[group][0] for group in selected_features] + \
                                     [self.group_dict[candidate][0]]
                    self.features = [item for sublist in features_group for item in sublist]
                    features = [item for sublist in features_group for item in sublist]
                    features.append('group_number')
                    self.X_train = self.featuresDF[features]
                    features_names = [self.group_dict[feature][1] for feature in selected_features] + \
                                     [self.group_dict[candidate][1]]

                print('{}: Start training with the groups: {} '.format((time.asctime(time.localtime(time.time()))),
                                                                       features_names))
                logging.info('{}: Start training with the groups: {} '
                             .format((time.asctime(time.localtime(time.time()))), features_names))
                group_results = self.models_iteration()
                best_auc = max(result[2] for result in group_results)
                auc_with_candidates.append((best_auc, candidate))

                print('{}: Finish training with the groups: {}'.
                      format((time.asctime(time.localtime(time.time()))), features_names))
                logging.info('{}: Finish training with the groups: {}'.
                             format((time.asctime(time.localtime(time.time()))), features_names))

                for model in group_results:
                    model.append(features_names)
                    model.append(opts.k_fold)
                columns_names = ['classifier_name', 'score', 'auc', 'train_time', 'features_list', 'k_fold']
                group_results_df = pd.DataFrame(group_results, columns=columns_names)
                # group_results.append(group_names).append([opts.k_fold])
                all_groups_results = all_groups_results.append(group_results_df, ignore_index=True)
                all_groups_results.to_csv('test_results_stepwise.csv', encoding='utf-8')

            auc_with_candidates.sort()
            best_new_auc, best_candidate = auc_with_candidates.pop()
            if current_auc <= best_new_auc:
                if opts.is_backward:  # use backward elimination
                    selected_features.remove(best_candidate)
                else:  # use forward selection
                    selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                current_auc = best_new_auc

            else:
                logging.info('{}: No candidate was chosen, number of selected features is {}.'.
                             format((time.asctime(time.localtime(time.time()))), len(selected_features)))
                print('{}: No candidate was chosen, number of selected features is {}.'.
                      format((time.asctime(time.localtime(time.time()))), len(selected_features)))

            # one candidate can be chosen, if not- we go forward to the next step.
            remain_number_of_candidate -= 1

        selected_features_names = [self.group_dict[feature][1] for feature in selected_features]
        logging.info('{}: Selected features are: {} and the best AUC is: {}'.
                     format((time.asctime(time.localtime(time.time()))), selected_features_names, best_new_auc))
        print('{}: Selected features for are: {} and the best AUC is: {}.'.
              format((time.asctime(time.localtime(time.time()))), selected_features_names, best_new_auc))

        return all_groups_results


###############################################################################
# benchmark classifiers
    def benchmark(self, clf, clf_name='default'):
        """
        This function train and test the model (clf) opts.k_fold time with CV
        :param clf: the model to train and test
        :param str clf_name: the name of the model
        :return: clf_name, average_acc, average_auc, train_time of the model
        :rtype list
        """
        print('_' * 80)
        print('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        logging.info('_' * 80)
        logging.info('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        # Cross validation part
        if clf_name == 'GaussianNB':
            self.X_train = self.X_train.toarray()
        t1 = time.time()
        score = []
        auc = []
        for out_group in range(1, opts.k_fold):
            t0 = time.time()
            # create train and test data
            test_data = self.X_train.loc[self.X_train['group_number'] == out_group, self.features]
            test_label = self.labels.loc[self.X_train['group_number'] == out_group, self.label_column_name]
            train_data = self.X_train.loc[self.X_train['group_number'] != out_group, self.features]
            train_label = self.labels.loc[self.X_train['group_number'] != out_group, self.label_column_name]

            # train the model
            clf.fit(train_data, train_label)
            predicted = clf.predict(test_data)
            score.append(metrics.accuracy_score(test_label, predicted))
            auc.append(metrics.roc_auc_score(test_label, predicted, average='samples'))

            logging.info("Fold number:")
            logging.info(out_group)
            logging.info("accuracy:")
            logging.info(metrics.accuracy_score(test_label, predicted))
            logging.info("AUC:")
            logging.info(metrics.roc_auc_score(test_label, predicted, average='samples'))
            if opts.print_cm:
                print("confusion matrix:")
                print(metrics.confusion_matrix(test_label, predicted, labels=[0, 1]))
                logging.info("confusion matrix:")
                logging.info(metrics.confusion_matrix(test_label, predicted, labels=[0, 1]))
            train_time = time.time() - t0
            # print("fold number {}: cross validation time: {}".format(out_group, train_time))
            logging.info("cross validation time: {}".format(train_time))

        # clf_descr = str(clf).split('(')[0]
        average_acc = sum(score)/len(score)
        print("Average Accuracy: {}".format(average_acc))
        logging.info("Average Accuracy: {})".format(average_acc))

        average_auc = sum(auc)/len(auc)
        print("Average AUC: {}".format(average_auc))
        logging.info('Average AUC: {}'.format(average_auc))

        train_time = time.time() - t1

        return [clf_name, average_acc, average_auc, train_time]

    def models_iteration(self):
        """
        This function iterate over model and call benchmark for each of them
        :return: list of lists with the results for each model
        """
        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(max_iter=1000), "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=1000), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(), 'Random forest'),  # n_estimators=100, bootstrap=False
                (SVC(C=1e-8, gamma=1.0/self.X_train.shape[1], kernel='rbf'), "SVM with RBF Kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf, name))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(loss='squared_hinge', penalty=penalty,
                                                    dual=False, tol=1e-3), 'LinearSVC_' + penalty))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=1000, penalty=penalty),
                                          'SGDClassifier_' + penalty))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=1000, penalty="elasticnet"),
                       "Elastic-Net penalty"))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid(), 'NearestCentroid'))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01), 'MultinomialNB'))
        results.append(self.benchmark(BernoulliNB(alpha=.01), 'BernoulliNB'))
        # results.append(self.benchmark(GaussianNB(), 'GaussianNB'))

        print('=' * 80)
        print("LinearSVC")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(LinearSVC(), 'LinearSVC'))

        print('=' * 80)
        print('Logistic Regression')
        results.append(self.benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                         intercept_scaling=1, penalty='l2', random_state=None,
                                                         tol=0.0001), 'Logistic Regression'))

        return results


if __name__ == '__main__':
    label_name = 'delta'
    classifier = Classifier(label_name)
    classifier.split_relevant_data()
    if opts.is_backward_forward:
        classifier_results = classifier.iterate_over_features_groups()
    else:
        classifier_results = classifier.create_data_no_feature_selection()

    classifier_results.to_csv(os.path.join(results_directory, 'CMV_classifier_results_all_deltas.csv'), encoding='utf-8')
