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
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from pdpbox import pdp
from rfpimp import *

# from pycebox.ice import ice, ice_plot

def df_string_to_cat(df:pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders


def df_cat_to_catcode(df):
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n//2
    nwomen = n//2
    df['ID'] = range(100,100+n)
    df['sex'] = ['M']*nmen + ['F']*nwomen
    df.loc[df['sex']=='F','pregnant'] = np.random.randint(0,2,size=(nwomen,))
    df.loc[df['sex']=='M','pregnant'] = 0
    df.loc[df['sex']=='M','height'] = 5*12+8 + np.random.uniform(-7, +8, size=(nmen,))
    df.loc[df['sex']=='F','height'] = 5*12+5 + np.random.uniform(-4.5, +5, size=(nwomen,))
    df.loc[df['sex']=='M','education'] = 10 + np.random.randint(0,8,size=nmen)
    df.loc[df['sex']=='F','education'] = 12 + np.random.randint(0,8,size=nwomen)
    df['weight'] = 120 \
                   + (df['height']-df['height'].min()) * 10 \
                   + df['pregnant']*10 \
                   - df['education']*1.2
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    return df


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


def ICE_predict(model, X:pd.DataFrame, colname:str, targetname="target"):
    """
    Return dataframe with one row per observation in X and one column
    per unique value of column identified by colname.
    Row 0 is actually the unique X[colname] values used to get predictions.
    It's handy to have so we don't have to pass X around to other methods.
    Points in a single ICE line are the unique values of colname zipped
    with one row of returned dataframe.
    """
    save = X[colname].copy()
    lines = np.zeros(shape=(len(X)+1, len(X[colname].unique())))
    uniq_values = sorted(X[colname].unique())
    lines[0,:] = uniq_values
    i = 0
    for v in uniq_values:
    #     print(f"{colname}.{v}")
        X[colname] = v
        y_pred = model.predict(X)
    #     print(y_pred)
        lines[1:,i] = y_pred
        i += 1
    X[colname] = save
    columns = [f"predicted {targetname}\n{colname}={str(v)}"
               for v in sorted(X[colname].unique())]
    return pd.DataFrame(lines, columns=columns)


def ICE_lines(ice:np.ndarray) -> np.ndarray:
    """
    Return a 3D array of 2D matrices holding X coordinates in col 0 and
    Y coordinates in col 1. result[0] is first 2D matrix of [X,Y] points
    in a single ICE line for single sample. Shape of result is:
    (nsamples,nuniquevalues,2)
    """
    linex = ice.iloc[0,:] # get unique x values from first row
    lines = []
    for i in range(1,len(ice)): # ignore first row
        liney = ice.iloc[i].values
        line = np.array(list(zip(linex, liney)))
        lines.append(line)
    return np.array(lines)


def plot_ICE(ax, ice, colname, targetname="target", linewidth=.7,
             color='#9CD1E3', alpha=.1, title=None,
             pdp=True, pdp_linewidth=1, pdp_alpha=1, pdp_color='black'):
    avg_y = np.mean(ice[1:], axis=0)
    min_pdp_y = avg_y[0]
    lines = ICE_lines(ice)
    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    # lines[:,:,0] scans all lines, all points in a line, and gets x column
    minx, maxx = np.min(lines[:,:,0]), np.max(lines[:,:,0])
    miny, maxy = np.min(lines[:,:,1]), np.max(lines[:,:,1])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)
    lines = LineCollection(lines, linewidth=linewidth, alpha=alpha, color=color)
    ax.add_collection(lines)
    if pdp:
        uniq_values = ice.iloc[0,:]
        ax.plot(uniq_values, avg_y - min_pdp_y,
                alpha=pdp_alpha, linewidth=pdp_linewidth, c=pdp_color)

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
    # print(f"leaf_samples {stop - start:.3f}s")
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


def partial_plot(ax, X, y, colname, targetname,
                 ntrees=30, min_samples_leaf=7,
                 numx=150,
                 alpha=.05,
                 xrange=None, yrange=None):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_models, leaf_ranges = piecewise_linear_leaves(rf, X, y, colname)
    minx = np.min(leaf_ranges[:, 0])
    maxx = np.max(leaf_ranges[:, 1])
    # print(f"range {minx:.3f}..{maxx:.3f}")
    # print(f"X[{colname}] range {np.min(X[colname])}..{np.max(X[colname])}")
    overall_range = np.array([minx, maxx])
    overall_axis = np.linspace(minx, maxx, num=numx)
    avg_at_x = curve_through_leaf_models(leaf_models, leaf_ranges, overall_axis)

    # Use OLS to determine hp and wgt relationship with mpg
    r = LinearRegression()
    r.fit(X, y)
    print(f"Regression on y~{list(X.columns.values)} predicting {targetname}")
    print(f"{targetname} = {r.coef_}*{list(X.columns.values)} + {r.intercept_}")

    # Use regr line to figure out how to get reliable left edge. RF edges are
    # fuzzy and that makes it impossible to just use avg_at_x[0] as min_y_at_left_edge_x
    # for centering
    r_curve = LinearRegression()
    r_curve.fit(overall_axis.reshape(-1,1), avg_at_x)
    ci = X.columns.get_loc(colname)
    print(f"Compare beta_{ci} = {r.coef_[ci]} to slope of avg curve {r_curve.coef_[0]}")

    min_y_at_left_edge_x = r_curve.predict(np.array(minx).reshape(-1,1))

    ax.scatter(overall_axis, avg_at_x - min_y_at_left_edge_x, s=2, alpha=1, c='black', label="Avg piecewise linear")
    segments = []
    miny = 9e10
    maxy = -9e10
    for r, lm in zip(leaf_ranges, leaf_models):
        rx = np.linspace(r[0], r[1], num=2) # just need endpoints for a line
        ry = lm.predict(rx.reshape(-1, 1)) - min_y_at_left_edge_x
        miny = min(miny, np.min(ry))
        maxy = max(maxy, np.max(ry))
        one_line = [(rx[0],ry[0]), (rx[1],ry[1])]
        segments.append( one_line )
        # segments.append(np.column_stack([r,ry]))
        # ax.plot(rx, ry, alpha=alpha, c='#9CD1E3')

    lines = LineCollection(segments, alpha=alpha, color='#9CD1E3')
    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(float(minx), float(maxx))
    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
        ax.set_ylim(miny, maxy)
    ax.add_collection(lines)
    # print("after all line segments")

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
    df_rent = df_rent.sample(n=1000)
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(4, 1, figsize=(6,16))
    partial_plot(axes[0], X, y, 'bedrooms', 'price')
    partial_plot(axes[1], X, y, 'bathrooms', 'price')
    partial_plot(axes[2], X, y, 'latitude', 'price')
    partial_plot(axes[3], X, y, 'longitude', 'price')
    plt.show()


def weight():
    df = toy_weight_data(200)
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(4, 2, figsize=(8,16))
    partial_plot(axes[0][0], X, y, 'education', 'weight')
    partial_plot(axes[1][0], X, y, 'height', 'weight')
    partial_plot(axes[2][0], X, y, 'sex', 'weight', yrange=(-40,40))
    partial_plot(axes[3][0], X, y, 'pregnant', 'weight', xrange=(0,4), yrange=(-20,20))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'education', 'weight')
    plot_ICE(axes[0,1], ice, 'education', 'weight')
    ice = ICE_predict(rf, X, 'height', 'weight')
    plot_ICE(axes[1,1], ice, 'height', 'weight')
    ice = ICE_predict(rf, X, 'sex', 'weight')
    plot_ICE(axes[2,1], ice, 'sex', 'weight')
    ice = ICE_predict(rf, X, 'pregnant', 'weight')
    plot_ICE(axes[3,1], ice, 'pregnant', 'weight')

    # pip install pycebox
    # I = ice(data=X, column='education', predict=rf.predict, num_grid_points=100)
    # ice_plot(I, ax=axes[0][1], plot_pdp=True, c='dimgray', linewidth=0.3)
    # I = ice(data=X, column='height', predict=rf.predict, num_grid_points=100)
    # ice_plot(I, ax=axes[1][1], plot_pdp=True, c='dimgray', linewidth=0.3)
    # I = ice(data=X, column='sex', predict=rf.predict, num_grid_points=2)
    # ice_plot(I, ax=axes[2][1], plot_pdp=True, c='dimgray', linewidth=0.3)
    # I = ice(data=X, column='pregnant', predict=rf.predict, num_grid_points=2)
    # ice_plot(I, ax=axes[3][1], plot_pdp=True, c='dimgray', linewidth=0.3)
    # y0 = I.T.iloc[:,0].values
    # y1 = I.T.iloc[:,1].values
    # print(y0)
    # print(y1)
    # segments = []
    # for y0_,y1_ in zip(y0,y1):
    #     segments.append( [(0,y0_), (1,y1_)] )
    # lines = LineCollection(segments, alpha=0.1, color='#9CD1E3')
    # axes[3][1].set_xlim(0,1)
    # axes[3][1].set_ylim(min(y0),max(y1))
    # axes[3][1].add_collection(lines)

    if False:
        # show importance as RF-piecewise linear plot see it
        rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, oob_score=True)
        rf.fit(X, y)

        I = importances(rf, X, y)
        plot_importances(I, ax=axes[4,0])
        axes[4, 0].set_title("Permutation importance")

        I = dropcol_importances(rf, X, y)
        plot_importances(I, ax=axes[4,1])
        axes[4, 1].set_title("Drop-column importance")

    # pip install pdpbox


    if False:
        p = pdp.pdp_isolate(rf, X, model_features=X.columns, feature='education')
        fig2, axes2 = \
            pdp.pdp_plot(p, 'education', plot_lines=True,
                         cluster=False,
                         n_cluster_centers=None)
        p = pdp.pdp_isolate(rf, X, model_features=X.columns, feature='height')
        fig2, axes2 = \
            pdp.pdp_plot(p, 'height', plot_lines=True,
                         cluster=False,
                         n_cluster_centers=None)

        p = pdp.pdp_isolate(rf, X, model_features=X.columns, feature='sex')
        fig2, axes2 = \
            pdp.pdp_plot(p, 'sex', plot_lines=True,
                         cluster=False,
                         n_cluster_centers=None)

        p = pdp.pdp_isolate(rf, X, model_features=X.columns, feature='pregnant')
        fig2, axes2 = \
            pdp.pdp_plot(p, 'pregnant', plot_lines=True,
                         cluster=False,
                         n_cluster_centers=None)

    plt.show()

if __name__ == '__main__':
    # cars()
    # rent()
    weight()