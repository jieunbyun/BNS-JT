from itertools import groupby
import logging
#import configparser
import pandas as pd

#from pathlib import Path


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# read node data
def read_nodes(file_node):

    df = pd.read_csv(file_node, index_col=0).to_dict('index')

    return {k: (v['x'], v['y']) for k, v in df.items()}


