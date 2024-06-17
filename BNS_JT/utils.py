from itertools import groupby
import logging
#import configparser
import pandas as pd
import numpy as np
from scipy import stats

#from pathlib import Path


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# read node data
def read_nodes(file_node):

    df = pd.read_csv(file_node, index_col=0).to_dict('index')

    return {k: (v['x'], v['y']) for k, v in df.items()}


def get_rat_dist(k1, k2, q, a1, a2, b1, b2, p):
    """
    Computes distribution of a ratio of (k1 + q1*X1) / (k1 + k2 + q*(X1+X2)),
    where X1~Beta(a1, b1) and X2~beta(a2,b2)

    Output:
    - mean: a float; mean value
    - std: a float; standard deviation
    - interval: a tuple; [0.5p, 1-0.5p] confidence interval

    Personal note: Given up deriving analytical form of the distribution. Future research?
    """
    # Check if all input values are positive
    if not all(i >= 0 for i in [k1, k2, a1, a2, b1, b2]):
        raise ValueError("All input values must be non-negative")

    if not (p > 0 and p < 1):
        raise ValueError("p (confidence interval prob.) must be between 0 and 1.")

    n_samp = int(1e6) # many samples
    # Generate samples
    x1 = np.random.beta(a=a1, b=b1, size=n_samp)
    x2 = np.random.beta(a=a2, b=b2, size=n_samp)

    # Calculate the mean difference and standard deviation of the difference
    ratio = (q*x1 + k1) / (q*(x1 + x2) + k1 + k2)
    mean_rat = np.mean(ratio)
    std_rat = np.std(ratio, ddof=1)

    # Calculate the 95% confidence interval for the difference
    ci_lower = np.percentile(ratio, 50*(1 - p))
    ci_upper = np.percentile(ratio, 100-50*(1 - p))

    return mean_rat, std_rat, (ci_lower, ci_upper)

