# Abstrat class for datasets, including the following methods:
# get_loaders()

from argparse import Namespace

class DatasetBase:
    NAME: str
    NUM_SAMPLES: int

    def __init__(self, **kwargs):
        """
        Base class for datasets.

        Args:
            dataset: the dataset at hand
            args: the arguments of the current execution
        """
        pass
        
    def get_loaders(self):
        """
        Get the loaders for training and testing.
        """
        raise NotImplementedError('get_loaders not implemented.')