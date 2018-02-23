"""
This script contains functions to produce a table showing the reduction in
imbalance of key variables across treatment and control groups before and
after matching, where matching is used to select a subset of the observations.
The code is based on the balance assessment routines described in Morgan
and Winship. 2015. Counterfactuals and Causal Inference. 2nd edn. p. 165-166.
Running this script will execute a simple example and print the results.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
from copy import copy
from collections import defaultdict

base_directory = os.path.abspath(os.curdir)
features_directory = os.path.join(base_directory, 'features_results')
imbalance_directory = os.path.join(base_directory, 'imbalance_results')


def standardized_diff_means(u, v):
    """
    Calculates the standardized difference in means between two numpy arrays.
    """
    return (np.mean(u) - np.mean(v)) / np.sqrt(0.5 * (np.var(u) + np.var(v)))


def log_ratio(treated, control):
    """
    Calculates the logarithm of the ratio of the STD of two numpy arrays.
    """
    std_treated = np.std(treated)
    std_control = np.std(control)
    ratio = std_treated / std_control
    return np.log(ratio)


class Imbalance:
    def __init__(self, data, treatment_column, matched_column, variable_names, treatments_list):
        self.treatment = dict()
        self.control = dict()
        self.treatment_matched = dict()
        self.control_matched = dict()
        self.covariates = copy(data[variable_names])
        self.matched = copy(data[matched_column])
        self.total_imbalance_unmatched = dict()
        self.total_imbalance_matched = dict()
        self.results = defaultdict(dict)
        # self.treatments_list = treatments_list
        self.variable_names = variable_names
        for treat in self.treatments_list:
            self.treatment[treat] = data.loc[data[treatment_column] == treat]
            self.control[treat] = data.loc[data[treatment_column] != treat]
            self.treatment_matched[treat] = data.loc[data[treatment_column] == treat & data[matched_column] == 'True']
            self.control_matched[treat] = data.loc[data[treatment_column] != treat & data[matched_column] == 'True']
            self.total_imbalance_unmatched[treat] = 0
            self.total_imbalance_matched[treat] = 0

    def imbalance(self, with_match=False):
        """
        Constructs a table showing the magnitude of imbalance between treated
        and control groups for unmatched and matched samples. The table includes
        the absolute standardized differences in sample means and p-values for each
        variable of interest before and after matching, as well as the overall
        level of bias.
        Parameters
        ----------
        :param: with_match: bool
            do we want to calculate also the matched
        Returns
        -------
        :return 2 pandas dataframe
            A dataframe containing information about the imbalance across treated
            and control units before and after matching for each of the treatments
        """
        for treat in self.treatments_list:
            for v in self.variable_names:
                # calculate measures
                treated = self.treatment[treat][v]
                control = self.control[treat][v]
                unmatched_diff = standardized_diff_means(treated, control)
                unmatched_log_ratio = log_ratio(treated, control)

                # create dict for treat
                self.results[treat][v] = {'unmatched treated mean': np.mean(treated),
                                          'unmatched control mean': np.mean(control),
                                          'unmatched treated STD': np.std(treated),
                                          'unmatched control STD': np.std(control),
                                          'unmatched standardized difference': unmatched_diff,
                                          'unmatched p-value': stats.ttest_ind(treated, control)[1],
                                          'unmatched log ratio': unmatched_log_ratio
                                          }

                # calculate the imbalance for all covariates
                self.total_imbalance_unmatched[treat] += unmatched_diff

                if with_match:  # calculate the matched parameters
                    # calculate the matched parameters
                    treated_matched = self.treatment_matched[treat][v]
                    control_matched = self.control_matched[treat][v]
                    matched_diff = standardized_diff_means(treated_matched, control_matched)
                    matched_log_ratio = log_ratio(treated_matched, control_matched)
                    # calculate the imbalance for all covariates
                    self.total_imbalance_matched[treat] += matched_diff
                    # add the matched parameters to the dict 1
                    self.results[treat][v]['matched treated mean'] = np.mean(treated_matched)
                    self.results[treat][v]['matched control mean'] = np.mean(control_matched)
                    self.results[treat][v]['matched treated STD'] = np.std(treated_matched)
                    self.results[treat][v]['matched control STD'] = np.std(control_matched)
                    self.results[treat][v]['matched log ratio'] = matched_log_ratio
                    self.results[treat][v]['matched difference'] = matched_diff
                    self.results[treat][v]['matched p-value'] = stats.ttest_ind(treated_matched, control_matched)[1]
                    self.results[treat][v]['% reduction in imbalance'] = (unmatched_diff / matched_diff - 1) * 100

            # calculate the imbalance for all covariates
            self.total_imbalance_unmatched[treat] /= len(self.variable_names)
            self.results[treat]['total_' + treat] = {'unmatched difference': self.total_imbalance_unmatched[treat]}

            if with_match:
                self.total_imbalance_matched[treat] /= len(self.variable_names)
                self.results[treat]['total_' + treat]['matched difference'] = self.total_imbalance_matched[treat]
                self.results[treat]['total_' + treat]['% reduction in imbalance'] =\
                    (self.total_imbalance_unmatched[treat] / self.total_imbalance_matched[treat] - 1) * 100

            # create total for treat 1
            results_df = pd.DataFrame.from_dict(self.results[treat], orient='index')
            results_df.to_csv(os.path.join(imbalance_directory, 'treat_' + treat + '_results.csv'))

        return


if __name__ == '__main__':
    features_data = pd.read_excel(os.path.join(features_directory, 'Features_causality_final.xlsx'))
    treatments = ['positive', 'negative']
    treatment_column_name = 'treated'
    matched_column_name = 'matched'
    variable_name = ['comment_author_number_original_subreddit', 'comment_author_number_recommend_subreddit',
                     'percent_efficient_references_comment_author', 'number_of_references_comment_author',
                     'number_of_efficient_references_comment_author', 'submission_author_number_original_subreddit',
                     'number_of_inefficient_references_comment_author', 'subreddits_similarity',
                     'submission_author_number_recommend_subreddit', 'cosine_similarity_subreddits_list',
                     'comment_created_time_hour', 'submission_created_time_hour', 'time_between_messages',
                     'comment_len', 'number_of_r', 'comment_submission_similarity', 'comment_title_similarity',
                     'number_of_references_to_submission', 'number_of_references_to_recommended_subreddit']
    imbalnce_obj = Imbalance(features_data, treatment_column_name, matched_column_name, variable_name, treatments)
    imbalnce_obj.imbalance()
