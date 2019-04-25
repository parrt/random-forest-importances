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


def scramble(X : np.ndarray) -> np.ndarray:
    """
    From Breiman: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    "...the first coordinate is sampled from the N values {x(1,n)}. The second
    coordinate is sampled independently from the N values {x(2,n)}, and so forth."
    """
    X_rand = X.copy()
    ncols = X.shape[1]
    for col in range(ncols):
        X_rand[:,col] = np.random.choice(np.unique(X[:,col]), len(X), replace=True)
    return X_rand


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


def conjure_twoclass(X):
    """
    Make new data set 2x as big with X and scrambled version of it that
    destroys structure between features. Old is class 0, scrambled is class 1.
    """
    if isinstance(X, pd.DataFrame):
        X_rand = df_scramble(X)
        X_synth = pd.concat([X, X_rand], axis=0)
    else:
        X_rand = scramble(X)
        X_synth = np.concatenate([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X)),
                              np.ones(len(X_rand))], axis=0)
    return X_synth, y_synth


# Derived from dtreeviz
def leaf_samples(tree_model, X):
    """
    Return dictionary mapping node id to list of sample indexes in leaf nodes.
    """
    tree = tree_model.tree_
    children_left = tree.children_left
    children_right = tree.children_right

    # Doc say: "Return a node indicator matrix where non zero elements
    #           indicates that the samples goes through the nodes."
    dec_paths = tree_model.decision_path(X)

    # each sample has path taken down tree
    node_to_leaves = defaultdict(list)
    for sample_i, dec in enumerate(dec_paths):
        _, nz_nodes = dec.nonzero()
        for node_id in nz_nodes:
            if children_left[node_id] == -1 and \
               children_right[node_id] == -1:  # is leaf?
                node_to_leaves[node_id].append(sample_i)

    return node_to_leaves


def wine():
    wine = load_wine()


def cars():
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG', 'WGT']]
    y = df_cars['MPG']
    ntrees = 20
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=5, oob_score=True)
    rf.fit(X[['WGT']], y)
    print(f"OOB R^2 {rf.oob_score_:.5f}")

    leaf_models = []
    leaf_ranges = []
    for tree in rf.estimators_:
        leaves = leaf_samples(tree, X[['WGT']])
        for leaf,samples in leaves.items():
            if len(samples)<2:
                print(f"ignoring len {len(samples)} leaf")
            leaf_hp = X.iloc[samples]['ENG']
            leaf_y = y.iloc[samples]
            lm = LinearRegression()
            lm.fit(leaf_hp.values.reshape(-1,1), leaf_y)
            leaf_models.append( lm )
            leaf_ranges.append( (min(leaf_hp), max(leaf_hp)) )

    step = 1
    leaf_ranges = np.array(leaf_ranges)
    minx = np.min(leaf_ranges[:,0])
    maxx = np.max(leaf_ranges[:,1])
    overall_range = np.array([minx, maxx])
    print(overall_range)
    overall_axis = np.arange(minx, maxx, step)
    curve = np.zeros(shape=(len(overall_axis), len(leaf_models)), dtype=np.float64)
    fig,axis = plt.subplots()
    i = 0 # leaf index; we get a line for each
    for r,lm in zip(leaf_ranges,leaf_models):
        rx = np.arange(r[0],r[1],step,dtype=int)
        ry = lm.predict(rx.reshape(-1,1))
        axis.plot(rx, ry, alpha=.1, c='#D9E6F5')

        # now save full range with 0s outside or r range
        ry = lm.predict(overall_axis.reshape(-1,1))
        ry[np.where(overall_axis<r[0])] = 0
        ry[np.where(overall_axis>r[1])] = 0
        curve[:,i] = ry
        i += 1

    sum_at_x = np.sum(curve, axis=1)
    count_at_x = np.count_nonzero(curve, axis=1)
    avg_at_x = sum_at_x / count_at_x
    # print(avg_at_x)

    axis.scatter(overall_axis, avg_at_x, s=2, alpha=1, c='black')
    # lm = LinearRegression()
    # lm.fit(overall_axis, leaf_hp.values.reshape(-1,1), leaf_y)

    # Use OLS to determine hp and wgt relationship with mpg
    r = LinearRegression()
    r.fit(X, y)
    print("Regression on hp,wgt predicting mpg")
    print(f"mpg = {r.coef_}*[hp wgt] + {r.intercept_}")
    axis.plot(overall_range,  overall_range*r.coef_[0],  linewidth=1, c='#fdae61')

    axis.set_xlabel("Horsepower")
    axis.set_ylabel("MPG")
    axis.set_title("Effect of HP on MPG in regions of similar car weights")
    plt.show()


def conjure_twoclass(X):
    X_rand = df_scramble(X)
    X_synth = pd.concat([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X)),
                              np.ones(len(X_rand))], axis=0)
    return X_synth, y_synth


if __name__ == '__main__':
    cars()