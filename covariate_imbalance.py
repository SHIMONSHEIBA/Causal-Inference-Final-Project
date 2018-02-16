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


def imbalance(df, variable_names, treatment_column, matched_column, treat1, treat2, with_match=False):
    """
    Constructs a table showing the magnitude of imbalance between treated
    and control groups for unmatched and matched samples. The table includes
    the absolute standardized differences in sample means and p-values for each
    variable of interest before and after matching, as well as the overall
    level of bias.
    Parameters
    ----------
    :param df : pandas DataFrame
        A dataframe containing all data used, as well as boolean columns
        indicating whether each observation is treated or control and whether
        each observation is matched or unmatched.
    :param variable_names : list of strings
        A list of strings corresponding to the columns for each we want to
        examine imbalance.
    :param treatment_column : string
        The name of the column containing treatment status.
    :param matched_column : string
        The name of the column indicating whether a variable is matched.
    :param: treat1 : string
        The first treatment we check, for example :positive
    :param: treat2 : string
        The second treatment we check, for example :negative
    :param: with_match: bool
        do we want to calculate also the matched
    Returns
    -------
    :return 2 pandas dataframe
        A dataframe containing information about the imbalance across treated
        and control units before and after matching for each of the treatments
    """
    results_1 = dict()
    results_2 = dict()
    total_imbalance_unmatched_1 = 0
    total_imbalance_matched_1 = 0
    total_imbalance_unmatched_2 = 0
    total_imbalance_matched_2 = 0
    for v in variable_names:
        # create relevant data
        treated_1 = df.loc[df[treatment_column] == treat1][v]
        control_1 = df.loc[df[treatment_column] != treat1][v]
        treated_2 = df.loc[df[treatment_column] == treat2][v]
        control_2 = df.loc[df[treatment_column] != treat2][v]

        # calculate measures
        unmatched_diff_1 = standardized_diff_means(treated_1, control_1)
        unmatched_diff_2 = standardized_diff_means(treated_2, control_2)
        unmatched_log_ratio_1 = log_ratio(treated_1, control_1)
        unmatched_log_ratio_2 = log_ratio(treated_2, control_2)

        # create dict for treat1
        results_1[v] = {'unmatched treated mean': np.mean(treated_1),
                        'unmatched control mean': np.mean(control_1),
                        'unmatched treated STD': np.std(treated_1),
                        'unmatched control STD': np.std(control_1),
                        'unmatched difference': unmatched_diff_1,
                        'unmatched p-value': stats.ttest_ind(treated_1, control_1)[1],
                        'unmatched log ratio': unmatched_log_ratio_1
                        }

        # create dict for treat1
        results_2[v] = {'unmatched treated mean': np.mean(treated_2),
                        'unmatched control mean': np.mean(control_2),
                        'unmatched treated STD': np.std(treated_2),
                        'unmatched control STD': np.std(control_2),
                        'unmatched difference': unmatched_diff_2,
                        'unmatched p-value': stats.ttest_ind(treated_2, control_2)[1],
                        'unmatched log ratio': unmatched_log_ratio_2
                        }

        # calculate the imbalance for all covariates
        total_imbalance_unmatched_1 += unmatched_diff_1
        total_imbalance_unmatched_2 += unmatched_diff_2

        if with_match:  # calculate the matched parameters
            # calculate the matched parameters
            treated_1_matched = df.loc[df[treatment_column] == treat1 & df[matched_column] == 'True'][v]
            control_1_matched = df.loc[df[treatment_column] != treat1 & df[matched_column] == 'True'][v]
            treated_2_matched = df.loc[df[treatment_column] == treat2 & df[matched_column] == 'True'][v]
            control_2_matched = df.loc[df[treatment_column] != treat2 & df[matched_column] == 'True'][v]
            matched_diff_1 = standardized_diff_means(treated_1_matched, control_1_matched)
            matched_diff_2 = standardized_diff_means(treated_2_matched, control_2_matched)
            matched_log_ratio_1 = log_ratio(treated_1_matched, control_1_matched)
            matched_log_ratio_2 = log_ratio(treated_2_matched, control_2_matched)
            # calculate the imbalance for all covariates
            total_imbalance_matched_1 += matched_diff_1
            total_imbalance_matched_2 += matched_diff_2
            # add the matched parameters to the dict 1
            results_1[v]['matched treated mean'] = np.mean(treated_1_matched)
            results_1[v]['matched control mean'] = np.mean(control_1_matched)
            results_1[v]['matched treated STD'] = np.std(treated_1_matched)
            results_1[v]['matched control STD'] = np.std(control_1_matched)
            results_1[v]['matched log ratio'] = matched_log_ratio_1
            results_1[v]['matched difference'] = matched_diff_1
            results_1[v]['matched p-value'] = stats.ttest_ind(treated_1_matched, control_1_matched)[1]
            results_1[v]['% reduction in imbalance'] = (unmatched_diff_1 / matched_diff_1 - 1) * 100
            # add the matched parameters to the dict 2
            results_2[v]['matched treated mean'] = np.mean(treated_2_matched)
            results_2[v]['matched control mean'] = np.mean(control_2_matched)
            results_2[v]['matched treated STD'] = np.std(treated_2_matched)
            results_2[v]['matched control STD'] = np.std(control_2_matched)
            results_2[v]['matched log ratio'] = matched_log_ratio_2
            results_2[v]['matched difference'] = matched_diff_2
            results_2[v]['matched p-value'] = stats.ttest_ind(treated_2_matched, control_2_matched)[1]
            results_2[v]['% reduction in imbalance'] = (unmatched_diff_2 / matched_diff_2 - 1) * 100

    # calculate the imbalance for all covariates
    total_imbalance_unmatched_1 /= len(variable_names)
    results_1['total_' + treat1] = {'unmatched difference': total_imbalance_unmatched_1}
    total_imbalance_unmatched_2 /= len(variable_names)
    results_2['total_' + treat2] = {'unmatched difference': total_imbalance_unmatched_2}

    if with_match:
        total_imbalance_matched_1 /= len(variable_names)
        results_1['total_' + treat1]['matched difference'] = total_imbalance_matched_1
        results_1['total_' + treat1]['% reduction in imbalance'] = (total_imbalance_unmatched_1 /
                                                                    total_imbalance_matched_1 - 1) * 100
        total_imbalance_matched_2 /= len(variable_names)
        results_2['total_' + treat2]['matched difference'] = total_imbalance_matched_2
        results_2['total_' + treat2]['% reduction in imbalance'] = (total_imbalance_unmatched_2 /
                                                                    total_imbalance_matched_2 - 1) * 100

    # create total for treat 1
    results_1_df = pd.DataFrame.from_dict(results_1, orient='index')

    # create total for treat 2
    results_2_df = pd.DataFrame.from_dict(results_2, orient='index')

    return results_1_df, results_2_df


if __name__ == '__main__':
    features_data = pd.read_excel(os.path.join(features_directory, 'example.xlsx'))
    treat_1 = 'positive'
    treat_2 = 'negative'
    variable_name = ['comment_author_number_original_subreddit', 'comment_author_number_recommend_subreddit',
                     'percent_efficient_references_comment_author', 'number_of_references_comment_author',
                     'number_of_efficient_references_comment_author', 'submission_author_number_original_subreddit',
                     'number_of_inefficient_references_comment_author', 'subreddits_similarity',
                     'submission_author_number_recommend_subreddit', 'cosine_similarity_subreddits_list',
                     'comment_created_time_hour', 'submission_created_time_hour', 'time_between_messages',
                     'comment_len', 'number_of_r', 'comment_submission_similarity', 'comment_title_similarity',
                     'number_of_references_to_submission', 'number_of_references_to_recommended_subreddit']
    result_1, result_2 = imbalance(features_data, variable_name, 'treated', 'matched', treat1=treat_1, treat2=treat_2)
    # save the results:
    result_1.to_csv(os.path.join(imbalance_directory, 'treat_' + treat_1 + '_results.csv'))
    result_2.to_csv(os.path.join(imbalance_directory, 'treat_' + treat_2 + '_results.csv'))
    print(result_1)
    print(result_2)
