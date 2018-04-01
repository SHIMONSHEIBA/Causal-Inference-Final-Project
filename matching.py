"""
This module contains a class to ensure that there is overlap between treatment and control with matching method.
"""
import numpy as np
import pandas as pd
import os


class Matching:

    def __init__(self):

        self.base_directory = os.path.abspath(os.curdir)
        self.data = pd.DataFrame()

        return

    def load_prepare_data(self, treatments_list, treatment_column,sub_directory, data_name, prepare = False):
        """
        This method loads the data and prepares binary columns for each value of the treatment feature
        :param treatments_list: all possible values of treatment feature
        :param treatment_column: name of treatment column in data
        :param data_name: data file name
        :return:
        """
        data = os.path.join(self.base_directory, sub_directory, data_name)
        if data_name[-3:] == 'csv':
            self.data.from_csv(data)
        else:
            self.data = pd.read_excel(data)

        if prepare:
            for treat in treatments_list:
                self.data[treat[0]] = np.where(self.data[treatment_column] == treat[0], 1, 0)
                #print('number of treatments for: {}, is: treat: {}, non treat')
        return

    def matching(self, label, propensity, calipher=0.05, replace=True):
        """ Performs nearest-neighbour matching for a sample of test and control
        observations, based on the propensity scores for each observation.

        Arguments:
        ----------
            label:      Series that contains the label for each observation.
            propensity: Series that contains the propensity score for each observation.
            calipher:   Bound on distance between observations in terms of propensity score.
            replace:    Boolean that indicates whether sampling is with (True) or without replacement (False).
        """

        treated = propensity[label == 1]
        control = propensity[label == 0]

        # Randomly permute in case of sampling without replacement to remove any bias arising from the
        # ordering of the data set
        matching_order = np.random.permutation(label[label == 1].index)
        matches = {}

        for obs in matching_order:
            # Compute the distance between the treatment observation and all candidate controls in terms of
            # propensity score
            distance = abs(treated[obs] - control)

            # Take the closest match
            if distance.min() <= calipher or not calipher:
                matches[obs] = [distance.argmin()]

                # Remove the matched control from the set of candidate controls in case of sampling without replacement
                if not replace:
                    control = control.drop(matches[obs])

        return matches


    def matching_to_dataframe(self, match, covariates, remove_duplicates=False):
        """ Converts a list of matches obtained from matching() to a DataFrame.
        Duplicate rows are controls that where matched multiple times.

        Arguments:
        ----------
            match:              Dictionary with a list of matched control observations.
            covariates:         DataFrame that contains the covariates for the observations.
            remove_duplicates:  Boolean that indicates whether or not to remove duplicate rows from the result.
                                If matching with replacement was used you should set this to False****
        """
        treated = list(match.keys())
        control = [ctrl for matched_list in match.values() for ctrl in matched_list]

        result = pd.concat([covariates.loc[treated], covariates.loc[control]])

        if remove_duplicates:
            return result.groupby(result.index).first()
        else:
            return result


    def trim_common_support(self, data, label_name, propensity_column_name):
        """ Removes observations that fall outside the common support of the propensity score
            distribution from the data.
            works with more than binary treatment


        Arguments:
        ----------
            data:        DataFrame with the propensity scores for each observation.
            label_name:  Column name that contains the labels (treatment/control) for each observation.

        """
        print('original data size is: {}'.format(data.shape))
        group_min_max = (data.groupby(label_name)
                         [propensity_column_name].agg({"min_propensity": np.min, "max_propensity": np.max}))

        # Compute boundaries of common support between the two propensity score distributions
        min_common_support = np.max(group_min_max.min_propensity)
        max_common_support = np.min(group_min_max.max_propensity)

        common_support = (data[propensity_column_name] >= min_common_support) & (data[propensity_column_name]
                                                                                 <= max_common_support)
        control = (data[label_name] == 0)
        print('before tream size of control for {} is {}'.format(label_name,control.sum()))
        treated = (data[label_name] == 1)
        print('before tream size of treated for {} is {}'.format(label_name, treated.sum()))

        treamed_data_points = common_support[common_support == False].index.tolist()
        print("treamed data points of total {} indexes are: {}".format(len(treamed_data_points), treamed_data_points))
        #TODO: check why when more than binary treatment, obs with propensity that is equal to limits
        # min/max do not return

        # Plot the resulting propensity score distribution to make sure we restricted ourselves to the common support
        print('propensity distribution of treatment and control after common support trim:')
        #TODO: fix saving plot
        # data[common_support].groupby(label_name)[propensity_column_name].plot(kind="hist",
        #                     sharex=True, range=(0, 1), bins=20, alpha=0.75).savefig('trimmed'+label_name+
        #                                                                                 propensity_column_name+'.png')
        print('treamed data size is: {}'.format(data[common_support].shape))
        return data[common_support]


def main():

    matching = Matching()
    treatments_list = [['treated','features_CMV.csv','propensity']]
    sub_directory = 'importing_change_my_view'
    prepare_data = False

    for treats in treatments_list:
        treatment_column = treats[0]
        data_name = treats[1]
        propensity_column_name = treats[2]
        # load data and create treatment columns
        matching.load_prepare_data(treatments_list, treatment_column, sub_directory, data_name, prepare_data)

        # following second assumption of common support, leave only data points that has probability greater than
        # zero to be both treatment/control
        common_support = matching.trim_common_support(matching.data, treatment_column, propensity_column_name)

        control = (common_support[treatment_column] == 0)
        print('after tream size of control for {} is {}'.format(treatment_column,control.sum()))
        treated = (common_support[treatment_column] == 1)
        print('after tream size of treated for {} is {}'.format(treatment_column, treated.sum()))

        matches = matching.matching(label=common_support[treatment_column],
                       propensity=common_support[propensity_column_name],
                       calipher=0.09,
                       replace=False)

        print('check if everybody got a match returns : {}'.format(sum([True if match == []
                                                                        else False for match in matches]) == 0))

        # return to df with all covariates
        matches_data_frame = matching.matching_to_dataframe(match=matches,
                                                   covariates=common_support,
                                                   remove_duplicates=False)
        # save matched df
        print('matched data size for {} is {}'.format(treatment_column,matches_data_frame.shape))
        matches_data_frame.to_csv('matches_data_frame_'+treatment_column+'_'+propensity_column_name+'.csv')

if __name__ == '__main__':
    main()
