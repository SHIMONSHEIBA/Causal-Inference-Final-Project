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

base_directory = os.path.abspath(os.curdir)
features_directory = os.path.join(base_directory, 'features_results')
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
            self.treatment[treat] = np.where(data[treatment_column] == treat, 1, 0)
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
            # TODO: dave the model's summary

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
    features_data = pd.read_excel(os.path.join(features_directory, 'Features_causality_final.xlsx'))
    treatments_list = ['positive', 'negative']
    treatment_column_main = 'treated'
    variable_name = ['comment_author_number_original_subreddit', 'comment_author_number_recommend_subreddit',
                     'percent_efficient_references_comment_author', 'number_of_references_comment_author',
                     'number_of_efficient_references_comment_author', 'submission_author_number_original_subreddit',
                     'number_of_inefficient_references_comment_author', 'subreddits_similarity',
                     'submission_author_number_recommend_subreddit', 'cosine_similarity_subreddits_list',
                     'comment_created_time_hour', 'submission_created_time_hour', 'time_between_messages',
                     'comment_len', 'number_of_r', 'comment_submission_similarity', 'comment_title_similarity',
                     'number_of_references_to_submission', 'number_of_references_to_recommended_subreddit']
    propensity_class = PropensityScore(data=features_data, variable_names=variable_name,
                                       treatment_column=treatment_column_main, treatments_list=treatments_list)

    for treatment in treatments_list:
        linear_formula = treatment + '~ ' + '+'.join(variable_name)
        for method in ['logistic', 'probit', 'linear']:
            column_name = 'propensity_score_' + treatment + '_' + method
            propensity_class.data_dict[treatment][column_name] =\
                propensity_class.estimate_propensity(treatment_name=treatment, method=method, formula=linear_formula)
            propensity_class.plot_propensity_hist(treatment=treatment, propensity_column_name=column_name)
        propensity_class.data_dict[treatment].to_csv(os.path.join(propensity_directory,
                                                                  'propensity_score_' + treatment + '.csv'))


if __name__ == '__main__':
    main()
