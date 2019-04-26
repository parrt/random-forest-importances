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
from  matplotlib.collections import LineCollection
import time


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
    start = time.time()
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

    stop = time.time()
    print(f"leaf_samples {stop - start:.3f}s")
    return node_to_leaves


def conjure_twoclass(X):
    X_rand = df_scramble(X)
    X_synth = pd.concat([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X)),
                              np.ones(len(X_rand))], axis=0)
    return X_synth, y_synth


def wine():
    wine = load_wine()


def piecewise_linear_leaves(rf, X, y, colname):
    start = time.time()
    leaf_models = []
    leaf_ranges = []
    for tree in rf.estimators_:
        leaves = leaf_samples(tree, X.drop(colname, axis=1))
        for leaf, samples in leaves.items():
            if len(samples) < 2:
                print(f"ignoring len {len(samples)} leaf")
            leaf_hp = X.iloc[samples][colname]
            leaf_y = y.iloc[samples]
            lm = LinearRegression()
            lm.fit(leaf_hp.values.reshape(-1, 1), leaf_y)
            leaf_models.append(lm)
            leaf_ranges.append((min(leaf_hp), max(leaf_hp)))
    leaf_ranges = np.array(leaf_ranges)
    stop = time.time()
    print(f"piecewise_linear_leaves {stop - start:.3f}s")
    return leaf_models, leaf_ranges


def curve_through_leaf_models(leaf_models, leaf_ranges, overall_axis):
    start = time.time()
    curve = np.zeros(shape=(len(overall_axis), len(leaf_models)), dtype=np.float64)
    i = 0  # leaf index; we get a line for each
    for r, lm in zip(leaf_ranges, leaf_models):
        # save full range with 0s outside or r range
        ry = lm.predict(overall_axis.reshape(-1, 1))
        ry[np.where(overall_axis < r[0])] = 0
        ry[np.where(overall_axis > r[1])] = 0
        curve[:, i] = ry
        i += 1
    sum_at_x = np.sum(curve, axis=1)
    count_at_x = np.count_nonzero(curve, axis=1)
    avg_at_x = sum_at_x / count_at_x
    stop = time.time()
    print(f"curve_through_leaf_models {stop - start:.3f}s")
    return avg_at_x


def lm_partial_plot(ax, X, y, colname, targetname):
    r_hp = LinearRegression()
    r_hp.fit(X[[colname]], y)
    print("\nRegression on hp predicting mpg")
    print(f"mpg_hp = {r_hp.coef_}*hp + {r_hp.intercept_}")
    r_wgt = LinearRegression()
    r_wgt.fit(X.drop(colname, axis=1), y)
    print("\nRegression on wgt predicting mpg")
    print(f"mpg_wgt = {r_wgt.coef_}*wgt + {r_wgt.intercept_}")
    ax.scatter(X[colname], y, alpha=.12)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(targetname+" vs "+colname)
    hp = X[colname]
    y_pred_hp = r_hp.predict(hp.values.reshape(-1, 1))
    ax.plot(hp, y_pred_hp, ":", linewidth=1, c='red', label='OLS y ~ ENG')
    r = LinearRegression()
    r.fit(X, y)
    xhp = np.linspace(min(hp), max(hp), num=100)
    ax.plot(xhp, xhp * r.coef_[0] + r_hp.intercept_, linewidth=1, c='orange', label="Beta_ENG")
    ax.legend()


def partial_plot(ax, X, y, colname, targetname, ntrees=20, min_samples_leaf=5, alpha=.05):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"OOB R^2 {rf.oob_score_:.5f}")
    leaf_models, leaf_ranges = piecewise_linear_leaves(rf, X, y, colname)
    numx = 100
    minx = np.min(leaf_ranges[:, 0])
    maxx = np.max(leaf_ranges[:, 1])
    # print(f"range {minx:.3f}..{maxx:.3f}")
    # print(f"X[{colname}] range {np.min(X[colname])}..{np.max(X[colname])}")
    overall_range = np.array([minx, maxx])
    overall_axis = np.linspace(minx, maxx, num=numx)
    avg_at_x = curve_through_leaf_models(leaf_models, leaf_ranges, overall_axis)
    min_y_at_left_edge_x = avg_at_x[0]

    ax.scatter(overall_axis, avg_at_x, s=2, alpha=1, c='black', label="Avg piecewise linear")
    print("after black curve")
    segments = []
    for r, lm in zip(leaf_ranges, leaf_models):
        rx = np.linspace(r[0], r[1], num=2) # just need endpoints for a line
        ry = lm.predict(rx.reshape(-1, 1))
        one_line = [(rx[0],ry[0]), (rx[1],ry[1])]
        segments.append( one_line )
        # segments.append(np.column_stack([r,ry]))
        #ax.plot(rx, ry, alpha=alpha, c='#9CD1E3')

    lines = LineCollection(segments)
    ax.add_collection(lines)
    print("after all line segments")

    # Use OLS to determine hp and wgt relationship with mpg
    r = LinearRegression()
    r.fit(X, y)
    ci = X.columns.get_loc(colname)
    print(f"Regression on y~{list(X.columns.values)} predicting {targetname}")
    print(f"{targetname} = {r.coef_}*{list(X.columns.values)} + {r.intercept_}")

    r_curve = LinearRegression()
    r_curve.fit(overall_axis.reshape(-1,1), avg_at_x)
    print(f"Compare beta_{ci} = {r.coef_[ci]} to slope of avg curve {r_curve.coef_[0]}")
    # ax.plot(overall_range, overall_range * r.coef_[ci], linewidth=1, c='#fdae61',
    #         label=f"Beta_{colname}={r.coef_[ci]}")
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(f"Effect of {colname} on {targetname} in similar regions")
    ax.legend()

    plt.tight_layout()


def cars():
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG', 'WGT']]
    y = df_cars['MPG']

    fig, axes = plt.subplots(2, 1, figsize=(5,6))
    lm_partial_plot(axes[0], X, y, 'ENG', 'MPG')
    partial_plot(axes[1], X, y, 'ENG', 'MPG')
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(5,6))
    lm_partial_plot(axes[0], X, y, 'WGT', 'MPG')
    partial_plot(axes[1], X, y, 'WGT', 'MPG')
    plt.show()


def rent():
    df_rent = pd.read_csv("/Users/parrt/github/mlbook-private/data/rent-ideal.csv")
    df_rent = df_rent.sample(n=100)
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(4, 1, figsize=(6,16))
    partial_plot(axes[0], X, y, 'bedrooms', 'price')
    partial_plot(axes[1], X, y, 'bathrooms', 'price')
    partial_plot(axes[2], X, y, 'latitude', 'price')
    partial_plot(axes[3], X, y, 'longitude', 'price')
    plt.show()


if __name__ == '__main__':
    # cars()
    rent()