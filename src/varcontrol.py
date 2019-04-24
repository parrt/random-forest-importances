import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata


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
            pass


def df_scramble(X : pd.DataFrame) -> pd.DataFrame:
    """
    From Breiman: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    "...the first coordinate is sampled from the N values {x(1,n)}. The second
    coordinate is sampled independently from the N values {x(2,n)}, and so forth."
    """
    X_rand = X.copy()
    for colname in X:
        X_rand[colname] = np.random.choice(X[colname].unique(), len(X), replace=True)
    return X_rand


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


def wine():
    wine = load_wine()


def cars():
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG', 'WGT']]
    y = df_cars['MPG']
    hp = X['ENG']
    wgt = X['WGT']
    # Make new data set 2x as big with X and scrambled version of it
    # that destroys structure between features. Old is class 0, scrambled
    # is class 1.
    X_synth, y_synth = conjure_twoclass(X)
    # Fit an RF on this new concat'd dataset
    rf = RandomForestRegressor(n_estimators=1, min_samples_leaf=5, oob_score=True)
    rf.fit(X_synth, y_synth)
    print(f"OOB R^2 {rf.oob_score_:.5f}")
    foo(rf, X, y)


def conjure_twoclass(X):
    X_rand = df_scramble(X)
    X_synth = pd.concat([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X)),
                              np.ones(len(X_rand))], axis=0)
    return X_synth, y_synth


if __name__ == '__main__':
    wine()