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
        self.regressor = LogisticRegression

        return

    def estimate_ate(self):
        #TODO: fit the regressor, add to normalized sums: 1. subtraction of y(i) - f(x(i),0) for t(i)=1
        #TODO: 2. f(x(i),1) -y(i) for t(i)=1 or 0?
        return