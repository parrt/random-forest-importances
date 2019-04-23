import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

"""
Play with controlling for other variables to see contribution of some x to y
"""

def foo(rf, X, y):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not hasattr(rf, 'estimators_'):  # make sure model is fit
        rf.fit(X, y)

    for t in rf.estimators_:
        nnodes = t.tree_.node_count
        left = t.tree_.children_left
        right = t.tree_.children_right
        print(nnodes)
        for n in range(nnodes):



# From dtreeviz
def node_samples(tree_model, X) -> Mapping[int, list]:
    """
    Return dictionary mapping node id to list of sample indexes considered by
    the feature/split decision.
    """
    # Doc say: "Return a node indicator matrix where non zero elements
    #           indicates that the samples goes through the nodes."
    dec_paths = tree_model.decision_path(X)

    # each sample has path taken down tree
    node_to_samples = defaultdict(list)
    for sample_i, dec in enumerate(dec_paths):
        _, nz_nodes = dec.nonzero()
        for node_id in nz_nodes:
            node_to_samples[node_id].append(sample_i)

    return node_to_samples


if __name__ == '__main__':
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG','WGT']]
    y = df_cars['MPG']

    hp = X['ENG']
    wgt = X['WGT']

    rf = RandomForestRegressor(n_estimators=1, min_samples_leaf=5)
    rf.fit(X, y)

    foo(rf, X, y)