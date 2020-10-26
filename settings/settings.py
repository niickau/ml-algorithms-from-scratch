# coding:utf-8
import random

from scipy.spatial.distance import euclidean, cosine, correlation
from collections import OrderedDict


class KD_Node:
    """Generic class for nodes for KD-Tree algorithm."""
    
    def __init__(self, depth=0):
        self.depth = depth
        self.data = None
        self.divide_dimension = None
        self.left = None
        self.right = None

def check_argument(arg):
    if not isinstance(arg, np.ndarray):
        arg_format = arg.to_numpy()
        return arg_format
    elif np.isnan(arg).any():
        raise ValueError("Matrix has NaN.")

def get_sub_indices(sub_matrix, main_matrix):
    indices = []
    for row in sub_matrix:
        indices.append(np.argmax((main_matrix == row).sum(axis=1)))
    return indices
