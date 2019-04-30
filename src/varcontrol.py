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
from scipy.integrate import cumtrapz

"""
Weakness: if model is bad minus feature x then plot for x is meaningless as RF
can't do a grouping into similar buckets. in contrast, PDP uses all x to make
predictions.

Good:

* Only considers points where we have data; then we use linear model to
  approximate the partial derivate. plot is then the integration. 
  
* We aren't using model to compute points, only locally to get slopes.

* No nonsensical samples; not touching data

* LM works great if all numeric data; this works with categorical

* Seems to isolate partial dependencies better
  
"""

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


def toy_crisscross_data():
    df = pd.DataFrame()
    x = np.linspace(0, 10, num=50)
    df['x1'] = x * 1.2
    df['x2'] = -x * 1.2 + 12
    df['y'] = df['x1'] * df['x2']
    return df


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


def toy_weather_data():
    def temp(x): return np.sin((x+365/2)*(2*np.pi)/365)
    df = pd.DataFrame()
    df['dayofyear'] = range(1,365+1)
    df['state'] = np.random.choice(['CA','CO','AZ','WA'], len(df))
    df['temperature'] = temp(df['dayofyear'])
    df.loc[df['state']=='CA','temperature'] = df.loc[df['state']=='CA','temperature'] * 70 #+ np.random.uniform(-20,40,sum(df['state']=='CA'))
    df.loc[df['state']=='CO','temperature'] = df.loc[df['state']=='CO','temperature'] * 40 #+ np.random.uniform(-20,40,sum(df['state']=='CO'))
    df.loc[df['state']=='AZ','temperature'] = df.loc[df['state']=='AZ','temperature'] * 90 #+ np.random.uniform(-20,40,sum(df['state']=='AZ'))
    df.loc[df['state']=='WA','temperature'] = df.loc[df['state']=='WA','temperature'] * 60 #+ np.random.uniform(-20,40,sum(df['state']=='WA'))
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
             yrange=None,
             pdp=True, pdp_linewidth=1, pdp_alpha=1, pdp_color='black'):
    avg_y = np.mean(ice[1:], axis=0)
    min_pdp_y = avg_y[0]
    lines = ICE_lines(ice)
    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    # lines[:,:,0] scans all lines, all points in a line, and gets x column
    minx, maxx = np.min(lines[:,:,0]), np.max(lines[:,:,0])
    miny, maxy = np.min(lines[:,:,1]), np.max(lines[:,:,1])
    ax.set_xlim(minx, maxx)
    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
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
                continue
            leaf_x = X.iloc[samples][colname]
            leaf_y = y.iloc[samples]
            r = (min(leaf_x), max(leaf_x))
            if r[0]==r[1]:
                print(f"ignoring xleft=xright @ {r[0]}")
                continue
            lm = LinearRegression()
            lm.fit(leaf_x.values.reshape(-1, 1), leaf_y)
            leaf_models.append(lm)
            leaf_ranges.append(r)
    leaf_ranges = np.array(leaf_ranges)
    stop = time.time()
    print(f"piecewise_linear_leaves {stop - start:.3f}s")
    return leaf_models, leaf_ranges


def catwise_leaves(rf, X, y, colname):
    """
    Return a dataframe with the average y value for each category in each leaf.
    The index has the complete category list. The columns are the y avg values
    found in a single leaf. Each row represents a category level. E.g.,

                       leaf0       leaf1
        category
        1         166.430176  186.796956
        2         219.590349  176.448626
    """
    start = time.time()
    catcol = X[colname].astype('category').cat.as_ordered()
    cats = catcol.cat.categories
    leaf_histos = pd.DataFrame(index=cats)
    leaf_histos.index.name = 'category'
    ci = 0
    for tree in rf.estimators_:
        leaves = leaf_samples(tree, X.drop(colname, axis=1))
        for leaf, samples in leaves.items():
            leaf_X = X.iloc[samples][colname]
            leaf_y = y.iloc[samples]
            combined = pd.concat([leaf_X, leaf_y], axis=1)
            grouping = combined.groupby(colname)
            # print(combined)
            histo = grouping.mean()
            avg_of_first_cat = histo.iloc[0]
            # print(histo)
            #             print(histo - min_of_first_cat)
            if len(histo) < 2:
                #                 print(f"ignoring len {len(histo)} cat leaf")
                continue
            delta_per_cat = histo - avg_of_first_cat
            leaf_histos['leaf' + str(ci)] = delta_per_cat
            ci += 1

    # print(leaf_histos)
    leaf_histos.fillna(0, inplace=True) # needed for |cats|>2
    stop = time.time()
    print(f"catwise_leaves {stop - start:.3f}s")
    return leaf_histos


def slopes_from_leaf_models(leaf_models, leaf_ranges):
    uniq_x = set(leaf_ranges[:, 0]).union(set(leaf_ranges[:, 1]))
    uniq_x = sorted(uniq_x)
    slopes = np.zeros(shape=(len(uniq_x), len(leaf_models)))
    i = 0  # leaf index; we get a line for each
    for r, lm in zip(leaf_ranges, leaf_models):
        x = np.full(len(uniq_x), lm.coef_[0])
        x[np.where(uniq_x < r[0])] = 0
        x[np.where(uniq_x > r[1])] = 0
    #     print(f"{r} -> {x}")
        slopes[:, i] = x
        i += 1
    sum_at_x = np.sum(slopes, axis=1)
    count_at_x = np.count_nonzero(slopes, axis=1)
    avg_slope_at_x = sum_at_x / count_at_x
    return uniq_x, avg_slope_at_x


def old_curve_through_leaf_models(leaf_models, leaf_ranges, overall_axis):
    start = time.time()
    # TODO: should we create nan not zeroes? what about a valid 0 value?
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

"""
piecewise-linear
partial dependence plot

PL PDP RF

partial dependence

piecewise partial dependence

stratification via random-forest and aggregation via averaging of piecewise-linear models 
"""

def partial_plot(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=30,
                 min_samples_leaf=7,
                 alpha=.05,
                 xrange=None,
                 yrange=None,
                 show_derivative=False):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_models, leaf_ranges = piecewise_linear_leaves(rf, X, y, colname)
    uniq_x, avg_slope_at_x = \
        slopes_from_leaf_models(leaf_models, leaf_ranges)

    if ax is None:
        fig, ax = plt.subplots(1,1)

    curve = cumtrapz(avg_slope_at_x, x=uniq_x)  # we lose one value here
    curve = np.concatenate([np.array([0]), curve])  # make it 0

    ax.scatter(uniq_x, curve,
               s=2, alpha=1,
               c='black', label="Avg piecewise linear")

    segments = []
    for r, lm in zip(leaf_ranges, leaf_models):
        delta = lm.coef_[0] * (r[1] - r[0])
        closest_x_i = np.abs(uniq_x - r[0]).argmin()
        y_offset = curve[closest_x_i]
        one_line = [(r[0],y_offset), (r[1], y_offset+delta)]
        segments.append( one_line )

    lines = LineCollection(segments, alpha=alpha, color='#9CD1E3', linewidth=1)
    if xrange is not None:
        ax.set_xlim(*xrange)
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.add_collection(lines)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(f"Effect of {colname} on {targetname} in similar regions")

    if show_derivative:
        other = ax.twinx()
        other.set_ylabel("Partial derivative", fontdict={"color":'#f46d43'})
        other.plot(uniq_x, avg_slope_at_x, linewidth=1, c='#f46d43')

    plt.tight_layout()


def cat_partial_plot(X, y, colname, targetname,
                     ax=None,
                     cats=None,
                     ntrees=30, min_samples_leaf=7,
                     numx=150,
                     alpha=.05,
                     xrange=None, yrange=None):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos = catwise_leaves(rf, X, y, colname)
    sum_per_cat = np.sum(leaf_histos, axis=1)
    # TODO: should we create nan not zeroes? what about a valid 0 value?
    nonzero_count_per_cat = np.count_nonzero(leaf_histos, axis=1)
    avg_per_cat = np.divide(sum_per_cat, nonzero_count_per_cat, where=nonzero_count_per_cat!=0)

    print(avg_per_cat)

    barcontainer = ax.bar(x=leaf_histos.index.values, height=avg_per_cat,
                          tick_label=leaf_histos.index,
                          align='center')

    for rect in barcontainer.patches:
        rect.set_linewidth(.5)
        rect.set_edgecolor(GREY)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(f"Effect of {colname} on {targetname} in similar regions")

    if yrange is not None:
        ax.set_ylim(*yrange)

    plt.tight_layout()


def cars():
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG', 'WGT']]
    y = df_cars['MPG']

    fig, axes = plt.subplots(2, 1, figsize=(5,6))
    lm_partial_plot(axes[0], X, y, 'ENG', 'MPG')
    partial_plot(X, y, 'ENG', 'MPG', ax=axes[1])
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(5,6))
    lm_partial_plot(axes[0], X, y, 'WGT', 'MPG')
    partial_plot(axes[1], X, y, 'WGT', 'MPG')
    plt.show()


def rent():
    df_rent = pd.read_csv("/Users/parrt/github/mlbook-private/data/rent-ideal.csv")
    df_rent = df_rent.sample(n=2000)
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(4, 1, figsize=(6,16))
    partial_plot(X, y, 'bedrooms', 'price', ax=axes[0])
    partial_plot(X, y, 'bathrooms', 'price', ax=axes[1])
    partial_plot(X, y, 'latitude', 'price', ax=axes[2])
    partial_plot(X, y, 'longitude', 'price', ax=axes[3])
    plt.show()


def weight():
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(3, 2, figsize=(8,8), gridspec_kw = {'height_ratios':[.2,3,3]})

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    partial_plot(X, y, 'education', 'weight', ax=axes[1][0],
                 ntrees=30, min_samples_leaf=7, yrange=(-12,0))
    # partial_plot(X, y, 'education', 'weight', ntrees=20, min_samples_leaf=7, alpha=.2)
    partial_plot(X, y, 'height', 'weight', ax=axes[2][0], yrange=(0,160))
    # cat_partial_plot(axes[2][0], X, y, 'sex', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    # cat_partial_plot(axes[3][0], X, y, 'pregnant', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'education', 'weight')
    plot_ICE(axes[1,1], ice, 'education', 'weight', yrange=(-12,0))
    ice = ICE_predict(rf, X, 'height', 'weight')
    plot_ICE(axes[2,1], ice, 'height', 'weight', yrange=(0,160))
    ice = ICE_predict(rf, X, 'sex', 'weight')
    # plot_ICE(axes[2,1], ice, 'sex', 'weight', yrange=(0,2))
    # ice = ICE_predict(rf, X, 'pregnant', 'weight')
    # plot_ICE(axes[3,1], ice, 'pregnant', 'weight', yrange=(0,10))

    fig.suptitle("weight = 120 + 10*(height-min(height)) + 10*pregnant - 1.2*education", size=14)

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

    plt.tight_layout()

    plt.savefig("/tmp/t.svg")
    plt.show()

def weather():
    df_raw = toy_weather_data()
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    print(catencoders)
    df_cat_to_catcode(df)
    X = df.drop('temperature', axis=1)
    y = df['temperature']

    fig, axes = plt.subplots(3, 2, figsize=(8,8), gridspec_kw = {'height_ratios':[.2,3,3]})

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    partial_plot(X, y, 'dayofyear', 'temperature', ax=axes[1][0],
                 ntrees=50, min_samples_leaf=7)#, yrange=(-12,0))
    # partial_plot(X, y, 'education', 'weight', ntrees=20, min_samples_leaf=7, alpha=.2)
    cat_partial_plot(X, y, 'state', 'temperature', ax=axes[2][0])#, yrange=(0,160))
    # cat_partial_plot(axes[2][0], X, y, 'sex', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    # cat_partial_plot(axes[3][0], X, y, 'pregnant', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'dayofyear', 'temperature')
    plot_ICE(axes[1,1], ice, 'dayofyear', 'temperature')#, yrange=(-12,0))
    ice = ICE_predict(rf, X, 'state', 'temperature')
    plot_ICE(axes[2,1], ice, 'state', 'temperature')#, yrange=(-12,0))

    # fig.suptitle("weight = 120 + 10*(height-min(height)) + 10*pregnant - 1.2*education", size=14)
    plt.tight_layout()

    plt.savefig("/tmp/weather.svg")
    plt.show()

def interaction():
    df = toy_crisscross_data()
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(3, 2, figsize=(8,8))

    axes[0,0].plot(range(len(df)), df['x1'], label="x1")
    axes[0,0].plot(range(len(df)), df['x2'], label="x2")
    axes[0,0].plot(range(len(df)), df['y'], label="y")
    axes[0, 0].set_xlabel("df row index")
    axes[0, 0].set_ylabel("df value")
    axes[0, 0].legend()
    axes[0, 0].set_title("Raw data; y = x1 * x2\nx1 = 1.2i; x2 = -1.2i + 12 for i=range(0..10,n=50)")
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,1].axis('off')

    partial_plot(X, y, 'x1', 'y', ax=axes[1][0],
                 ntrees=50, min_samples_leaf=7, yrange=(0,40))
    # partial_plot(X, y, 'education', 'weight', ntrees=20, min_samples_leaf=7, alpha=.2)
    partial_plot(X, y, 'x2', 'y', ax=axes[2][0], yrange=(0,40))
    # cat_partial_plot(axes[2][0], X, y, 'sex', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    # cat_partial_plot(axes[3][0], X, y, 'pregnant', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'x1', 'y')
    plot_ICE(axes[1,1], ice, 'x1', 'y', yrange=(0,40))
    axes[1, 1].set_title("Partial dependence plot")
    ice = ICE_predict(rf, X, 'x2', 'y')
    plot_ICE(axes[2,1], ice, 'x2', 'y', yrange=(0,40))

    plt.tight_layout()

    plt.savefig("/tmp/weather.svg")
    plt.show()



if __name__ == '__main__':
    # cars()
    # rent()
    # weight()
    # weather()
    interaction()