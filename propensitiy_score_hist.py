"""
This module contains a class to estimate propensity scores.
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sma
import statsmodels.formula.api as smf
from copy import copy
import os
from sklearn import metrics

base_directory = os.path.abspath(os.curdir)
features_directory = os.path.join(base_directory, 'change my view')
propensity_directory = os.path.join(base_directory, 'propensity_score_results')


class PropensityScore(object):
    """
    Estimate the propensity score for each observation.
    """

    def __init__(self, data, treatment_column, variable_names, treatments_list):
        """
        :param data: A data frame containing all data used, as well as boolean columns
                    indicating whether each observation is treated or control and whether
                    each observation is matched or unmatched.
        :type data: pandas Data frame
        :param treatment_column: The name of the column containing treatment status.
        :type treatment_column: str
        :param variable_names: A list of strings corresponding to the columns of the features.
        :type variable_names: list(str)
        :param treatments_list: list of treatments we check, for example: ['positive', 'negative']
        :type treatments_list: list(str)
        """

        self.treatment = pd.DataFrame(copy(data[treatment_column]))
        self.covariates = copy(data[variable_names])
        self.data_dict = dict()
        for treat in treatments_list:
            # self.treatment[treat] = np.where(data[treatment_column] == treat, 1, 0)
            self.data_dict[treat] = pd.concat([self.treatment[treat], self.covariates], axis=1)

    def estimate_propensity(self, method, treatment_name, formula=None, show_summary=True):
        """
        Compute propensity score per measure and treatment
        :param method: Propensity score estimation method. Either 'logistic', 'probit', or 'linear'
        :type method: str
        :param formula: If method = 'linear': String that contains the (R style) model formula
        :type formula: str
        :param show_summary: indicates to print out the model summary to the console.
        :type show_summary: bool
        :param treatment_name: what is the treatment
        :type treatment_name: str
        :return: series with the predicted propensity score
        :rtype: pandas Series
        """

        data = sm.add_constant(self.data_dict[treatment_name], prepend=False)
        treatment = self.treatment[treatment_name]
        predictors = sm.add_constant(self.covariates, prepend=False)
        if method == 'logistic':
            model = sm.Logit(treatment, predictors).fit(disp=False, warn_convergence=True)
            predicted = pd.Series(data=model.predict(predictors), index=data.index)
        elif method == 'probit':
            model = sm.Probit(treatment, predictors).fit(disp=False, warn_convergence=True)
            predicted = pd.Series(data=model.predict(predictors), index=data.index)
        elif method == 'linear':
            model = smf.glm(formula=formula, data=data, family=sma.families.Binomial()).fit()
            predicted = pd.Series(data=model.predict(data), index=data.index)
        else:
            raise ValueError('Unrecognized method')

        if show_summary:
            print('Summary of the model: {} for treatment: {}'.format(method, treatment_name))
            print(model.summary())
            # TODO: save the model's summary

        return predicted

    def plot_propensity_hist(self, treatment, propensity_column_name):
        """
        Create histogram of propensity scores and save it to file
        :param treatment: what is the treatment
        :type treatment: str
        :param propensity_column_name: The name of the column containing propensity score.
        :type propensity_column_name: str
        """

        data = copy(self.data_dict[treatment])
        hist_title = 'Histogram_' + propensity_column_name
        plot = plt.figure(hist_title)
        plt.hist(data.loc[data[treatment] == 1][propensity_column_name], fc=(0, 0, 1, 0.5), bins=20, label='Treated')
        plt.hist(data.loc[data[treatment] == 0][propensity_column_name], fc=(1, 0, 0, 0.5), bins=20, label='Control')
        plt.legend()
        plt.title(hist_title)
        plt.xlabel('propensity score')
        plt.ylabel('number of units')
        fig_to_save = plot
        fig_to_save.savefig(os.path.join(propensity_directory, hist_title + '.png'), bbox_inches='tight')

        return


def main():
    features_data = pd.read_csv(os.path.join(features_directory, 'features_CMV.csv'))
    treatments_list = [1]
    treatment_column_main = 'treated'
    variable_name = ['commenter_number_submission', 'submitter_number_submission', 'is_first_comment_in_tree',
                     'number_of_comments_in_tree_from_submitter', 'number_of_respond_by_submitter_total',
                     'respond_to_comment_user_responses_ratio', 'submitter_seniority_days', 'nltk_com_sen_neg',
                     'nltk_sub_sen_neg', 'nltk_title_sen_neg', 'commenter_number_comment', 'submitter_number_comment',
                     'number_of_comments_in_tree_by_comment_user', 'number_of_respond_by_submitter',
                     'respond_to_comment_user_all_ratio', 'respond_total_ratio', 'commenter_seniority_days',
                     'nltk_com_sen_neutral', 'nltk_sub_sen_neutral', 'nltk_title_sen_neutral', 'topic_model'
                     'time_ratio', 'nltk_com_sen_pos', 'nltk_sub_sen_pos', 'nltk_title_sen_pos', 'comment_len'
                     'nltk_sim_sen', 'percent_adj', 'submission_len', 'title_len', 'time_between_messages',
                     'time_until_first_comment', 'time_between_comment_first_comment',
                     'submmiter_commenter_tfidf_cos_sim']
    y_column_name = 'delta'
    propensity_class = PropensityScore(data=features_data, variable_names=variable_name,
                                       treatment_column=treatment_column_main, treatments_list=treatments_list)

    for treatment in treatments_list:
        linear_formula = str(treatment) + '~ ' + '+'.join(variable_name)
        for method in ['logistic', 'probit', 'linear']:
            column_name = 'propensity_score_' + str(treatment) + '_' + method
            propensity_class.data_dict[treatment][column_name] =\
                propensity_class.estimate_propensity(treatment_name=str(treatment), method=method,
                                                     formula=linear_formula)
            propensity_class.plot_propensity_hist(treatment=str(treatment), propensity_column_name=column_name)
        data_to_save = pd.concat([propensity_class.data_dict[treatment],
                                  pd.DataFrame(copy(features_data[y_column_name]))], axis=1)
        data_to_save.to_csv(os.path.join(propensity_directory, 'CMV_propensity_score_' + str(treatment) + '.csv'))
        # estimate propensity method
        # print(metrics.roc_auc_score(test_label, propensity_class.data_dict[treatment][column_name], average='samples'))


if __name__ == '__main__':
    main()
