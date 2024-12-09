import numpy as np
import torch

from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError
from utils import create_if_not_exists
import os

import pickle


def append_dictionary(dic1, dic2):
    """
    Extend dictionary dic1 with dic2.
    """
    for key in dic2.keys():
        if key in dic1.keys():
            dic1[key].append(dic2[key])
        else:
            dic1[key] = [dic2[key]]


def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
        curr_ind = pred[:, curr_k]
        num_eq = torch.eq(curr_ind, target).sum()
        acc = num_eq / len(output)
        res_total += acc
    return res_total * 100


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
