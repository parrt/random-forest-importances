import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

    # Predict wgt using hp
    print("\nRegression predicting y from residuals of hp predicting wgt")
    r_hp_wgt = LinearRegression()
    r_hp_wgt.fit(X[['ENG']], X['WGT'])
    print(f"wgt = {r_hp_wgt.coef_}*hp + {r_hp_wgt.intercept_}")
    # Residual of true wgt might prediction based upon hp
    res_wgt = wgt - (r_hp_wgt.coef_ * hp + r_hp_wgt.intercept_)

    # Predict mpg using wgt using hp
    r_y_hp_res = LinearRegression()
    r_y_hp_res.fit(res_wgt.values.reshape(-1,1), y)
    print(f"Beta for wgt is {r_y_hp_res.coef_} + {r_y_hp_res.intercept_} (see mpg coeff[1])")

    # Predict hp using wgt
    print("\nRegression predicting y from residuals of wgt predicting hp")
    r_wgt_hp = LinearRegression()
    r_wgt_hp.fit(X[['WGT']], X['ENG'])
    print(f"hp = {r_wgt_hp.coef_}*wgt + {r_wgt_hp.intercept_}")
    # Residual of true wgt might prediction based upon hp
    res_hp  = hp  - (r_wgt_hp.coef_ * wgt + r_wgt_hp.intercept_)

    # Predict mpg using wgt using hp
    r_y_wgt_res = LinearRegression()
    r_y_wgt_res.fit(res_hp.values.reshape(-1,1), y)
    print(f"Beta for hp is {r_y_wgt_res.coef_} + {r_y_wgt_res.intercept_}(see mpg coeff[0])")

    fig,axes = plt.subplots(1,2,sharey=True)
    axes[0].scatter(X['ENG'],y,alpha=.25)
    axes[1].scatter(X['WGT'],y,c='blue',alpha=.2)
    axes[0].set_xlabel('Horsepower')
    axes[1].set_xlabel('Weight')
    axes[0].set_ylabel('MPG')

    xhp = np.arange(min(hp),max(hp),1)
    xwgt = np.arange(min(wgt),max(wgt),1)

    # Show biased simple y=x regressions
    axes[0].plot(xhp, r_hp.predict(xhp.reshape(-1,1)), "-", linewidth=1, c='red')
    axes[1].plot(xwgt,r_wgt.predict(xwgt.reshape(-1,1)), "-", linewidth=1, c='red')

    # Show partial dependencies from regression coeff
    axes[0].plot(xhp,xhp*r.coef_[0]+r_hp.intercept_, linewidth=1, c='orange')
    axes[1].plot(xwgt,xwgt*r.coef_[1]+r_wgt.intercept_, linewidth=1, c='orange')

    plt.show()