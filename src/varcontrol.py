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

def foo(tree_model, X, y):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if getattr(tree_model, 'tree_') is None:  # make sure model is fit
        tree_model.fit(X, y)

    nnodes = tree_model.tree_.node_count
    left = tree_model.tree_.children_left
    right = tree_model.tree_.children_right

    for tree in tree_model.estimators_:
        print(tree)


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


def beta_hp(X, y):
    # Predict hp using wgt
    print("\nRegression predicting y from residuals of wgt predicting hp")
    r_wgt_hp = LinearRegression()
    r_wgt_hp.fit(X[['WGT']], X['ENG'])
    print(f"hp = {r_wgt_hp.coef_}*wgt + {r_wgt_hp.intercept_}")
    # Residual of true wgt might prediction based upon hp
    # res_hp is the variation in hp that can't be explained by wgt
    res_hp  = hp - (r_wgt_hp.coef_ * wgt + r_wgt_hp.intercept_)

    # Predict mpg using wgt using hp
    r_y_wgt_res = LinearRegression()
    r_y_wgt_res.fit(res_hp.values.reshape(-1,1), y)
    return r_y_wgt_res.coef_


def beta_hp_rf(X, y) -> RandomForestRegressor:
    # Predict hp using wgt
    rf_wgt_hp = RandomForestRegressor(n_estimators=100)
    rf_wgt_hp.fit(X[['WGT']], X['ENG'])
    # Residual of true wgt might prediction based upon hp
    # res_hp is the variation in hp that can't be explained by wgt
    res_hp  = hp - rf_wgt_hp.predict(wgt.values.reshape(-1,1))
    print(pd.concat([X,res_hp,y], axis=1).head(50))

    # Predict mpg using wgt using hp
    r_y_wgt_res = RandomForestRegressor(n_estimators=100)
    r_y_wgt_res.fit(res_hp.values.reshape(-1,1), y)
    # return res_hp.unique(), r_y_wgt_res.predict(res_hp.unique().reshape(-1,1))
    return res_hp, r_y_wgt_res.predict(res_hp.values.reshape(-1,1))


def beta_wgt(X,y):
    # Predict wgt using hp
    print("\nRegression predicting y from residuals of hp predicting wgt")
    r_hp_wgt = LinearRegression()
    r_hp_wgt.fit(X[['ENG']], X['WGT'])
    print(f"wgt = {r_hp_wgt.coef_}*hp + {r_hp_wgt.intercept_}")
    # Residual of true wgt might prediction based upon hp
    # res_wgt is the variation in wgt that can't be explained by hp
    res_wgt = wgt - (r_hp_wgt.coef_ * hp + r_hp_wgt.intercept_)

    # Predict mpg using wgt using hp
    rf_y_hp_res = LinearRegression()
    rf_y_hp_res.fit(res_wgt.values.reshape(-1,1), y)
    return rf_y_hp_res.coef_


def beta_wgt_rf(X,y) -> np.ndarray:
    # Predict wgt using hp
    rf_hp_wgt = RandomForestRegressor(n_estimators=100)
    rf_hp_wgt.fit(X[['ENG']], X['WGT'])
    # Residual of true wgt might prediction based upon hp
    # res_wgt is the variation in wgt that can't be explained by hp
    res_wgt = wgt - rf_hp_wgt.predict(hp.values.reshape(-1,1))

    # Predict mpg using wgt using hp
    rf_y_hp_res = RandomForestRegressor(n_estimators=100)
    rf_y_hp_res.fit(res_wgt.values.reshape(-1,1), y)
    # return res_wgt.unique(), rf_y_hp_res.predict(res_wgt.unique().reshape(-1,1))
    return res_wgt, rf_y_hp_res.predict(res_wgt.values.reshape(-1,1))


if __name__ == '__main__':
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG','WGT']]
    y = df_cars['MPG']

    hp = X['ENG']
    wgt = X['WGT']

    r = LinearRegression()
    r.fit(X, y)
    print("Regression on hp,wgt predicting mpg")
    print(f"mpg = {r.coef_}*[hp wgt] + {r.intercept_}")

    r_hp = LinearRegression()
    r_hp.fit(X[['ENG']], y)
    print("\nRegression on hp predicting mpg")
    print(f"mpg_hp = {r_hp.coef_}*hp + {r_hp.intercept_}")

    r_wgt = LinearRegression()
    r_wgt.fit(X[['WGT']], y)
    print("\nRegression on wgt predicting mpg")
    print(f"mpg_wgt = {r_wgt.coef_}*wgt + {r_wgt.intercept_}")

    print(f"Beta for hp is {beta_hp(X,y)} (see mpg coeff[1])")
    # print(f"Beta via RF for hp is {beta_hp_rf(X,y)} (see mpg coeff[1])")

    print(f"Beta for wgt is {beta_wgt(X,y)} (see mpg coeff[1])")
    # print(f"Beta via RF for wgt is {beta_wgt_rf(X,y)} (see mpg coeff[1])")

    fig,axes = plt.subplots(3,2,sharey=False)
    tl = axes[0,0]
    tr = axes[0,1]
    tl.scatter(X['ENG'],y,alpha=.25)
    tr.scatter(X['WGT'],y,c='blue',alpha=.2)
    tl.set_xlabel('Horsepower')
    tr.set_xlabel('Weight')
    tl.set_ylabel('MPG')

    xhp = np.arange(0,max(hp),1)
    xwgt = np.arange(0,max(wgt),1)
    xhp = np.arange(min(hp),max(hp),1)
    xwgt = np.arange(min(wgt),max(wgt),1)

    # Show biased simple y=x regressions
    tl.plot(xhp, r_hp.predict(xhp.reshape(-1,1)), ":", linewidth=1.5, c='black')
    tr.plot(xwgt,r_wgt.predict(xwgt.reshape(-1,1)), ":", linewidth=1.5, c='black')

    # Show RF derived partial dep
    res_hp,foo = beta_hp_rf(X, y)
    axes[1,0].scatter(X['ENG'],res_hp, linewidth=1, s=3, c='green')
    res_wgt,foo = beta_wgt_rf(X, y)
    axes[1,1].scatter(X['WGT'], res_wgt, linewidth=1, s=3, c='green')

    res_hp,foo = beta_hp_rf(X, y)
    axes[2,0].scatter(res_hp, y, linewidth=1, s=3, c='green')
    res_wgt,foo = beta_wgt_rf(X, y)
    axes[2,1].scatter(res_wgt, y, linewidth=1, s=3, c='green')

    # Show partial dependencies from regression coeff
    tl.plot(xhp,xhp*r.coef_[0], linewidth=1, c='orange')
    tr.plot(xwgt,xwgt*r.coef_[1], linewidth=1, c='orange')

    plt.show()