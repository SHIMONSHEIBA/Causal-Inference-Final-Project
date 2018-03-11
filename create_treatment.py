import pandas as pd
import os
import logging
from datetime import datetime
from copy import copy


base_directory = os.path.abspath(os.curdir)
change_my_view_directory = os.path.join(base_directory, 'change my view')
log_directory = os.path.join(base_directory, 'logs')
LOG_FILENAME = os.path.join(log_directory,
                            datetime.now().strftime('LogFile_create_treatment_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, )


class CreateTreatment:
    """
    This class build the treatment column - if there is a quote in the comment T=1
    """
    def __init__(self):
        self.data = pd.read_excel(os.path.join(change_my_view_directory, 'small_data.xlsx'))  # filter_comments_submissions
        self.data.assign(treated='')

    def loop_over_data(self):
        """
        Check if there is a quote in the comment.
        Check if it is a quote of a comment of the submitter or the submission itself
        :return:
        """
        for index, comment in self.data.iterrows():
            comment_body = copy(comment['comment_body'])
            is_quote = 0
            if '>' in comment_body:  # this is the sign for a quote (|) in the comment
                while '>' in comment_body and not is_quote:
                    quote_index = comment_body.find('>')
                    comment_body = comment_body[quote_index + 1:]
                    is_quote = self.check_quote(comment, comment_body, index)
            else:  # there is not quote at all
                self.data.loc[index, 'treated'] = 0

        return

    def check_quote(self, comment, comment_body, index):
        comment_id = comment['comment_id']
        if '>>>' in comment_body:
            print('There is >>> in comment_id: {}'.format(comment['comment_id']))
            logging.info('There is >>> in comment_id: {}'.format(comment['comment_id']))
        if '>>' in comment_body:  # this is the sign of a higher hierarchy - there might be another quote after it
            double_index = comment_body.find('>>')
            uni_index = comment_body.find('> ')
            if uni_index == -1:  # there is no > in the comment
                print('only >> and not > in comment_id: {}'.format(comment_id))
                logging.info('only >> and not > in comment_id: {}'.format(comment_id))
            elif uni_index < double_index:  # there is quote before the higher hierarchy quote
                comment_body = comment_body
            else:  # there is a quote after the higher hierarchy
                comment_body_short = comment_body[double_index + 2:]  # the body without the '>>'
                # if '>' in comment_body_short:  # there is another quote - we need it
                #     comment_body = copy(comment_body_short)
                #     uni_index = comment_body.find('>')
                #     comment_body = copy(comment_body[uni_index:])  # get the comment from the inner quote
                if '>' not in comment_body_short:
                    print('There is a >> but not > after it in comment_id: {}'.format(comment_id))
                    logging.info('There is a >> but not > after it in comment_id: {}'.format(comment_id))
        quote = copy(comment_body)
        nn_index = quote.find('\\n')
        n_index = quote.find('\n')
        if nn_index == -1:  # there is no \\n
            quote = quote
        elif n_index != -1:  # there is \n
            quote = quote[: n_index - 1]
        else:
            quote = quote[: nn_index - 1]  # take the quote: after the > and until the first \n
        # parse the parent id
        parent_id = comment['parent_id']
        if 't1_' in parent_id:
            parent_id = parent_id.lstrip('b').strip("'").lstrip('t1_')
        elif 't3_' in parent_id:
            parent_id = parent_id.lstrip('b').strip("'").lstrip('t3_')
        else:
            print('not t_ in parent_id for comment_id: {}'.format(comment_id))
            logging.info('not t_ in parent_id for comment_id: {}'.format(comment_id))

        # if the parent is the submission - take the submission body
        if parent_id == comment['submission_id']:
            parent_body = comment['submission_body']
            parent_author = comment['submission_author']
        else:  # if not - get the parent
            parent_id = "b'" + parent_id + "'"
            values = self.data.loc[self.data['comment_id'] == parent_id]
            if values.empty:  # if we don't have the parent as comment in the data
                print('no parent comment for comment_id: {}'.format(comment_id))
                logging.info('no parent comment for comment_id: {}'.format(comment_id))
                parent_body = ''
                parent_author = ''
            else:
                parent = pd.Series(values.iloc[0])
                parent_body = parent['comment_body']
                parent_author = parent['comment_author']
        submission_author = comment['submission_author']
        submission_body = comment['submission_body']
        submission_title = comment['submission_title']

        if submission_author == parent_author:  # check if the parent author is the submitter
            # if he quote the submission or the parent
            if (quote in parent_body) or (quote in submission_body) or (quote in submission_title):
                self.data.loc[index, 'treated'] = 1
                return 1
            else:  # he didn't quote the submitter
                self.data.loc[index, 'treated'] = 0
                return 0
        else:  # if the parent author is not the submitter
            if (quote in submission_body) or (quote in submission_title):  # we only care of he quote the submission:
                self.data.loc[index, 'treated'] = 1
                print('quote the submission, but it is not its parent for comment_id: {}'.format(comment_id))
                logging.info('quote the submission, but it is not its parent for comment_id: {}'.format(comment_id))
                return 1
            else:
                self.data.loc[index, 'treated'] = 0
                return 0


def main():
    treatment_object = CreateTreatment()
    treatment_object.loop_over_data()
    # writer = pd.ExcelWriter(os.path.join(change_my_view_directory, 'data_label_treatment.xlsx'))
    # treatment_object.data.to_excel(writer, 'data_label_treatment')
    # writer.save()
    treatment_object.data.to_csv(os.path.join(change_my_view_directory, 'data_label_treatment.csv'))


if __name__ == '__main__':
    main()
