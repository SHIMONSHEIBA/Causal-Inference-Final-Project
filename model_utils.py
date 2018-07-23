import torch as tr
import torch.utils.data as dt
import torchtext
import spacy


class CustomDataset(dt.Dataset):

    """
    Class for handling data before modeling, preprocess and loading
    """

    def __init__(self):
        """
        this method uploads and organize all elements in the data for the different parts of the model
        """
        #TODO: import data, prepare different elements,
        #TODO: pre process what is needed for all data and not for single data point

        # First part of dataset is the raw text, tensor M*N:
        # M - number of branches, N - maximum number of comments in branch, content - raw text

        # Second part of dataset is the features of the comments, tensor M*N, content - comment features

        # Third part of dataset is the profiles of the users, tensor M*N, content - user's features

        # Forth part of dataset contains two dictionaries:
        # 1. {branch index: submission id}
        # 2. {submission id: [submission text, submission features, submitter profile features]}

        # Forth part of the dataset is a dictionary: {branch index: [is delta in branch, number of deltas in branch,
        # [deltas comments location in branch]]}
        pass

    def __getitem__(self, index):
        """
        this method takes all elements of a single data point for onr iteration of the model, and returns them with
        it's label. the dataloader will use this method for taking batches in training.
        :param index: index of data point to be retrieved
        :return: (data point elements, label)
        """
        #TODO: take all elements of one data point preprocess it and return the data point and it's label
        x = tr.Tensor([[1,3],[2,0]])
        y = tr.Tensor([1,0])
        return x,y

    def __len__(self):
        """

        :return: size of dataset
        """
        # TODO: return size of dataset
        return 0