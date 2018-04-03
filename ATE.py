"""
This module contains methods to estimate ATE
"""
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
import statsmodels.api as sma
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression


class Ate:

    def __init__(self, data):

        self.data = data

        return

    def estimate_ate(self, Y, X, treatment_column, model):

        gold_treated = Y.loc[Y[treatment_column] == 1, "delta"]
        gold_control = Y.loc[Y[treatment_column] == 0, "delta"]
        num_of_treated = gold_treated.shape[0]
        num_of_control = gold_control.shape[0]

        predicted_treated_as_control_covariates = X.loc[X[treatment_column] == 1]
        predicted_treated_as_control_covariates[[treatment_column]] = 0
        predicted_control_as_treated_covariates = X.loc[X[treatment_column] == 0]
        predicted_control_as_treated_covariates[[treatment_column]] = 1

        predicted_treated_as_control = model.predict(predicted_treated_as_control_covariates)
        predicted_control_as_treated = model.predict(predicted_control_as_treated_covariates)

        part_one = (1/num_of_treated)*sum(gold_treated - predicted_treated_as_control)
        part_two = (1/num_of_control)*sum(predicted_control_as_treated - gold_control)
        covariate_adjustment = part_one + part_two

        return covariate_adjustment

def main():

    log_model = LogisticRegression()
    treatments_list = [['treated','features_CMV.csv']]
    sub_directory = 'importing_change_my_view'
    for treats in treatments_list:
        treatment_column = treats[0]
        data_name = treats[1]
        #propensity_column_name = treats[2]
        data_path = os.path.join(sub_directory, data_name)
        data = pd.read_csv(data_path)
        X = data.drop(['delta'], axis=1)
        Y = data[['delta',treatment_column]]
        log_model.fit(X=X,y=Y['delta'])
        ate = Ate(data)
        covariate_adjustment = ate.estimate_ate( Y, X, treatment_column, log_model)
        print("covariate_adjustment ATE estimate for {} treatment is: {}".format(treatment_column, covariate_adjustment))
    return


if __name__ == '__main__':
    main()