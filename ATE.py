"""
This module contains methods to estimate ATE
"""
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB


def estimate_ate(Y, X, treatment_column, model, label_column):
    """
    This function estimate the ATE for the treatment
    :param pandas DF Y: the labels
    :param pandas DF X: the covariates
    :param str treatment_column: the name of the treatment column
    :param model: the trained model
    :param str label_column: the name of the label column
    :return: the treatment effect based on the covarite adjustment method
    """

    gold_treated = Y.loc[Y[treatment_column] == 1, label_column]
    gold_control = Y.loc[Y[treatment_column] == 0, label_column]
    num_of_treated = gold_treated.shape[0]
    num_of_control = gold_control.shape[0]

    predicted_treated_as_control_covariates = X.loc[X[treatment_column] == 1]
    predicted_treated_as_control_covariates[[treatment_column]] = 0
    predicted_control_as_treated_covariates = X.loc[X[treatment_column] == 0]
    predicted_control_as_treated_covariates[[treatment_column]] = 1

    predicted_treated_as_control = model.predict(predicted_treated_as_control_covariates)
    predicted_control_as_treated = model.predict(predicted_control_as_treated_covariates)

    part_one = (1/num_of_treated)*sum(gold_treated - predicted_treated_as_control)
    print("part one is: ", part_one)
    part_two = (1/num_of_control)*sum(predicted_control_as_treated - gold_control)
    print("part two is: ", part_two)
    covariate_adjustment = part_one + part_two

    return covariate_adjustment


def estimate_ate2(Y, X, treatment_column, model, label_column, variable_name, propensity_name):
    """
    This function estimate the ATE for the treatment
    :param pandas DF Y: the labels
    :param pandas DF X: the covariates
    :param str treatment_column: the name of the treatment column
    :param model: the trained model
    :param str label_column: the name of the label column
    :return: the treatment effect based on the covarite adjustment method
    """

    propensity = X[propensity_name]
    X = X[variable_name]
    gold = Y[label_column]
    num_of_units = gold.shape[0]

    part_one = (1/num_of_units)*sum((gold * X[treatment_column])/propensity)
    print("part one2 is: ", part_one)
    part_two = (1 / num_of_units) * sum((gold * (1-X[treatment_column])) / (1-propensity))
    print("part two2 is: ", part_two)
    ATE = part_one + part_two

    return ATE


def main():

    model = MultinomialNB(alpha=.01)
    treatments_list = [['treated', 'delta', 'matches_data_frame_treated_propensity_score_treated_logistic.csv',
                        'propensity_score_treated_logistic']]
    variable_name = ['commenter_number_submission', 'submitter_number_submission',
                     'submitter_seniority_days', 'is_first_comment_in_tree',
                     'commenter_number_comment', 'submitter_number_comment',
                     'number_of_comments_in_tree_by_comment_user', 'commenter_seniority_days',
                     'time_ratio', 'comment_len', 'submission_len', 'title_len', 'time_between_messages',
                     'time_until_first_comment', 'time_between_comment_first_comment', 'comment_depth',
                     'number_of_comments_in_tree_from_submitter', 'number_of_respond_by_submitter_total',
                     'respond_to_comment_user_responses_ratio', 'number_of_respond_by_submitter',
                     'respond_to_comment_user_all_ratio', 'respond_total_ratio', 'treated', 'nltk_com_sen_pos',
                     'nltk_com_sen_neg', 'nltk_com_sen_neutral', 'nltk_sub_sen_pos', 'nltk_sub_sen_neg',
                     'nltk_sub_sen_neutral', 'nltk_title_sen_pos', 'nltk_title_sen_neg', 'nltk_title_sen_neutral',
                     'nltk_sim_sen', 'percent_adj', 'submmiter_commenter_tfidf_cos_sim',
                     'topic_model_0', 'topic_model_1', 'topic_model_2', 'topic_model_3', 'topic_model_4',
                     'topic_model_5', 'topic_model_6', 'topic_model_7', 'topic_model_8', 'topic_model_9',
                     'topic_model_10', 'topic_model_11', 'topic_model_12', 'topic_model_13', 'topic_model_14']
    sub_directory = 'importing_change_my_view'
    for treats in treatments_list:
        treatment_column = treats[0]
        data_name = treats[2]
        propensity_column_name = treats[3]
        data_path = os.path.join(sub_directory, data_name)
        data = pd.read_csv(data_path)
        X = data[variable_name]
        Y = data[[treats[1], treatment_column]]
        model.fit(X=X, y=Y[treats[1]])
        variable_name2 = variable_name + [propensity_column_name]
        X2 = data[variable_name2]
        covariate_adjustment = estimate_ate(Y, X, treatment_column, model, treats[1])
        covariate_adjustment2 = estimate_ate2(Y, X2, treatment_column, model, treats[1], variable_name,
                                              propensity_column_name)
        print("covariate_adjustment ATE estimate for {} treatment is: {}".format(treatment_column,
                                                                                 covariate_adjustment))
        print("ATE2 estimate for {} treatment is: {}".format(treatment_column, covariate_adjustment2))
    return


if __name__ == '__main__':
    main()