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
from dtreeviz.trees import *


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


def toy_crisscross_data(n=50):
    df = pd.DataFrame()
    i = np.linspace(0, 10, num=n)
    df['x1'] = i * 1.2
    df['x2'] = -i * 1.2 + 12
    df['y'] = df['x1'] * df['x2']# + df['x1'] + df['x2']
    return df, f"y = x1x2\nx1 = 1.2i; x2 = -1.2i + 12 for i=range(0..10,n={n})"


def toy_twolines_data(n=50):
    df = pd.DataFrame()
    i = np.linspace(0, 10, num=n)
    df['x1'] = i * 1
    df['x2'] = i * 2
    df['y'] = df['x1'] * df['x2']# + df['x1'] + df['x2']
    return df, f"y = x1x2\nx1 = i; x2 = 2i for i=range(0..10,n={n})"


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
    df.loc[df['state']=='CA','temperature'] = 70 + df.loc[df['state']=='CA','temperature'] * np.random.uniform(-20,40,sum(df['state']=='CA'))
    df.loc[df['state']=='CO','temperature'] = 40 + df.loc[df['state']=='CO','temperature'] * np.random.uniform(-20,40,sum(df['state']=='CO'))
    df.loc[df['state']=='AZ','temperature'] = 90 + df.loc[df['state']=='AZ','temperature'] * np.random.uniform(-20,40,sum(df['state']=='AZ'))
    df.loc[df['state']=='WA','temperature'] = 60 + df.loc[df['state']=='WA','temperature'] * np.random.uniform(-20,40,sum(df['state']=='WA'))
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


def ICE_predict(model, X:pd.DataFrame, colname:str, targetname="target", numx=None):
    """
    Return dataframe with one row per observation in X and one column
    per unique value of column identified by colname.
    Row 0 is actually the sorted unique X[colname] values used to get predictions.
    It's handy to have so we don't have to pass X around to other methods.
    Points in a single ICE line are the unique values of colname zipped
    with one row of returned dataframe. E.g.,

    	predicted weight          predicted weight         ...
    	height=62.3638789416112	  height=62.78667197542318 ...
    0	62.786672	              70.595222                ... unique X[colname] values
    1	109.270644	              161.270843               ...
    """
    save = X[colname].copy()
    if numx is not None:
        linex = np.linspace(np.min(X[colname]), np.max(X[colname]), numx)
    else:
        linex = sorted(X[colname].unique())
    lines = np.zeros(shape=(len(X) + 1, len(linex)))
    lines[0, :] = linex
    i = 0
    for v in linex:
        #         print(f"{colname}.{v}")
        X[colname] = v
        y_pred = model.predict(X)
        lines[1:, i] = y_pred
        i += 1
    X[colname] = save
    columns = [f"predicted {targetname}\n{colname}={str(v)}"
               for v in linex]
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


def plot_ICE(ice, colname, targetname="target", cats=None, ax=None, linewidth=.7, color='#9CD1E3',
             alpha=.1, title=None, yrange=None, pdp=True, pdp_linewidth=1, pdp_alpha=1,
             pdp_color='black'):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    avg_y = np.mean(ice[1:], axis=0)
    min_pdp_y = avg_y[0] if cats is None else np.min(avg_y)
    lines = ICE_lines(ice)
    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    # lines[:,:,0] scans all lines, all points in a line, and gets x column
    minx, maxx = np.min(lines[:,:,0]), np.max(lines[:,:,0])
    miny, maxy = np.min(lines[:,:,1]), np.max(lines[:,:,1])
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

    if cats is not None:
        ncats = len(cats)
        ax.set_xticks(range(0, ncats))
        ax.set_xticklabels(cats)
        ax.set_xlim(range(0, ncats))
    else:
        ax.set_xlim(minx, maxx)

    ax.set_title(f"Partial dependence of {colname} on {targetname}")

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


def collect_leaf_slopes(rf, X, y, colname):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform a regression to get the slope of X[colname] vs y.
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the range of X[colname] and associated slope.
    """
    start = time.time()
    leaf_slopes = []
    leaf_ranges = []
    for tree in rf.estimators_:
        leaves = leaf_samples(tree, X.drop(colname, axis=1))
        for leaf, samples in leaves.items():
            if len(samples) < 2:
                # print(f"ignoring len {len(samples)} leaf")
                continue
            leaf_x = X.iloc[samples][colname]
            leaf_y = y.iloc[samples]
            r = (min(leaf_x), max(leaf_x))
            if np.isclose(r[0], r[1]):
                # print(f"ignoring xleft=xright @ {r[0]}")
                continue
            lm = LinearRegression()
            lm.fit(leaf_x.values.reshape(-1, 1), leaf_y)
            leaf_slopes.append(lm.coef_[0])
            leaf_ranges.append(r)
    leaf_ranges = np.array(leaf_ranges)
    stop = time.time()
    # print(f"collect_leaf_slopes {stop - start:.3f}s")
    return leaf_ranges, np.array(leaf_slopes)


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
#             print("\n", combined)
            histo = grouping.mean()
#             print(grouping.count())
#             print(histo)
            #             print(histo - min_of_first_cat)
            if len(histo) < 2:
                #                 print(f"ignoring len {len(histo)} cat leaf")
                continue
            leaf_histos['leaf' + str(ci)] = histo
            ci += 1

    # print(leaf_histos)
    stop = time.time()
    print(f"catwise_leaves {stop - start:.3f}s")
    return leaf_histos


def avg_slope_at_x(leaf_ranges, leaf_slopes):
    uniq_x = set(leaf_ranges[:, 0]).union(set(leaf_ranges[:, 1]))
    uniq_x = np.array(sorted(uniq_x))
    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope in zip(leaf_ranges, leaf_slopes):
        s = np.full(nx, slope) # s has value scope at all locations (flat line)
        # now trim line so it's only valid in range r
        s[np.where(uniq_x < r[0])] = np.nan
        s[np.where(uniq_x > r[1])] = np.nan
        slopes[:, i] = s
        i += 1
    # Now average horiz across the matrix, averaging within each range
    sum_at_x = np.nansum(slopes, axis=1)
    missing_values_at_x = np.isnan(slopes).sum(axis=1)
    count_at_x = nslopes - missing_values_at_x
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    avg_slope_at_x = sum_at_x / count_at_x
    return uniq_x, avg_slope_at_x


def lm_partial_plot(X, y, colname, targetname,ax=None):
    r_col = LinearRegression()
    r_col.fit(X[[colname]], y)
    ax.scatter(X[colname], y, alpha=.12)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(targetname+" vs "+colname)
    col = X[colname]
    y_pred_hp = r_col.predict(col.values.reshape(-1, 1))
    ax.plot(col, y_pred_hp, ":", linewidth=1, c='red', label='OLS y ~ ENG')
    r = LinearRegression()
    r.fit(X, y)
    xhp = np.linspace(min(col), max(col), num=100)
    ci = X.columns.get_loc(colname)
    ax.plot(xhp, xhp * r.coef_[ci] + r_col.intercept_, linewidth=1, c='orange', label="Beta_ENG")
    left30 = xhp[int(len(xhp) * .3)]
    ax.text(left30, left30*r.coef_[ci] + r_col.intercept_, f"slope={r.coef_[ci]:.3f}")
    ax.legend()

def partial_plot(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=30,
                 min_samples_leaf=7,
                 alpha=.05,
                 xrange=None,
                 yrange=None,
                 show_derivative=True):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"\nModel wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_ranges, leaf_slopes = collect_leaf_slopes(rf, X, y, colname)
    uniq_x, slope_at_x = avg_slope_at_x(leaf_ranges, leaf_slopes)
    # print(f'uniq_x = [{", ".join([f"{x:4.1f}" for x in uniq_x])}]')
    # print(f'slopes = [{", ".join([f"{s:4.1f}" for s in slope_at_x])}]')

    if ax is None:
        fig, ax = plt.subplots(1,1)

    curve = cumtrapz(slope_at_x, x=uniq_x)          # we lose one value here
    curve = np.concatenate([np.array([0]), curve])  # add back the 0 we lost

    ax.scatter(uniq_x, curve,
               s=2, alpha=1,
               c='black', label="Avg piecewise linear")

    segments = []
    for r, slope in zip(leaf_ranges, leaf_slopes):
        delta = slope * (r[1] - r[0])
        closest_x_i = np.abs(uniq_x - r[0]).argmin()
        y_offset = curve[closest_x_i]
        one_line = [(r[0],y_offset), (r[1], y_offset+delta)]
        segments.append( one_line )

    lines = LineCollection(segments, alpha=alpha, color='#9CD1E3', linewidth=1)
    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(min(uniq_x),max(uniq_x))
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.add_collection(lines)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(f"Effect of {colname} on {targetname} in similar regions\nOOB R^2 {rf.oob_score_:.3f}")

    if show_derivative:
        other = ax.twinx()
        other.set_ylabel("Partial derivative", fontdict={"color":'#f46d43'})
        other.plot(uniq_x, slope_at_x, linewidth=1, c='#f46d43', alpha=.5)
        other.set_ylim(min(slope_at_x),max(slope_at_x))
        other.set_xlim(min(uniq_x),max(uniq_x))
        other.tick_params(axis='y', colors='#f46d43')
        m = np.mean(slope_at_x)
        mx =max(uniq_x)
        other.plot(mx-mx*0.03, m, marker='>', c='#f46d43')


def cat_partial_plot(X, y, colname, targetname,
                     cats=None,
                     ax=None,
                     sort='ascending',
                     ntrees=30, min_samples_leaf=7,
                     alpha=.03,
                     yrange=None):
    rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True, n_jobs=-1)
    rf.fit(X.drop(colname, axis=1), y)
    print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos = catwise_leaves(rf, X, y, colname)
    sum_per_cat = np.sum(leaf_histos, axis=1)
    nonmissing_count_per_cat = len(leaf_histos.columns) - np.isnan(leaf_histos).sum(axis=1)
    avg_per_cat = sum_per_cat / nonmissing_count_per_cat

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ncats = len(cats)
    nleaves = len(leaf_histos.columns)

    sort_indexes = range(ncats)
    if sort == 'ascending':
        sort_indexes = avg_per_cat.argsort()
        cats = cats[sort_indexes]
    elif sort == 'descending':
        sort_indexes = avg_per_cat.argsort()[::-1]  # reversed
        cats = cats[sort_indexes]

    min_value = np.min(avg_per_cat)

    xloc = 1
    sigma = .02
    mu = 0
    x_noise = np.random.normal(mu, sigma, size=nleaves)
    for i in sort_indexes:
        ax.scatter(x_noise + xloc, leaf_histos.iloc[i]-min_value,
                   alpha=alpha, marker='o', s=10,
                   c='#9CD1E3')
        ax.plot([xloc - .1, xloc + .1], [avg_per_cat.iloc[i]-min_value] * 2,
                c='black', linewidth=2)
        xloc += 1
    ax.set_xticks(range(1, ncats + 1))
    ax.set_xticklabels(cats)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    ax.set_title(f"Effect of {colname} on {targetname} in similar regions")

    if yrange is not None:
        ax.set_ylim(*yrange)


def cars():
    df_cars = pd.read_csv("/Users/parrt/github/dtreeviz/testing/data/cars.csv")
    X = df_cars[['ENG', 'WGT']]
    y = df_cars['MPG']

    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    lm_partial_plot(X, y, 'ENG', 'MPG', ax=axes[0,0])
    partial_plot(X, y, 'ENG', 'MPG', ax=axes[0,1], yrange=(-20,20))

    lm_partial_plot(X, y, 'WGT', 'MPG', ax=axes[1,0])
    partial_plot(X, y, 'WGT', 'MPG', ax=axes[1,1], yrange=(-20,20))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    ice = ICE_predict(rf, X, 'ENG', 'MPG')
    plot_ICE(ice, 'ENG', 'MPG', ax=axes[0, 2], yrange=(-20,20))
    ice = ICE_predict(rf, X, 'WGT', 'MPG')
    plot_ICE(ice, 'WGT', 'MPG', ax=axes[1, 2], yrange=(-20,20))

    plt.tight_layout()

    plt.show()


def rent():
    df_rent = pd.read_csv("/Users/parrt/github/mlbook-private/data/rent-ideal.csv")
    df_rent = df_rent.sample(n=1500)
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(4, 2, figsize=(8,16))
    partial_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], yrange=(0,3000))
    partial_plot(X, y, 'bathrooms', 'price', ax=axes[1,0], yrange=(0,5000))
    partial_plot(X, y, 'latitude', 'price', ax=axes[2,0], yrange=(0,1300))
    partial_plot(X, y, 'longitude', 'price', ax=axes[3,0], yrange=(-3000,250))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'bedrooms', 'price')
    plot_ICE(ice, 'bedrooms', 'price', ax=axes[0, 1], yrange=(0,3000))
    ice = ICE_predict(rf, X, 'bathrooms', 'price')
    plot_ICE(ice, 'bathrooms', 'price', ax=axes[1, 1])
    ice = ICE_predict(rf, X, 'latitude', 'price')
    plot_ICE(ice, 'latitude', 'price', ax=axes[2, 1], yrange=(0,1300))
    ice = ICE_predict(rf, X, 'longitude', 'price')
    plot_ICE(ice, 'longitude', 'price', ax=axes[3, 1], yrange=(-3000,250))

    plt.tight_layout()

    plt.show()


def weight():
    df_raw = toy_weight_data(200)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(5, 2, figsize=(8,16), gridspec_kw = {'height_ratios':[.2,3,3,3,3]})

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    partial_plot(X, y, 'education', 'weight', ax=axes[1][0],
                 ntrees=30, min_samples_leaf=7, yrange=(-12,0))
    partial_plot(X, y, 'height', 'weight', ax=axes[2][0], yrange=(0,160))
    cat_partial_plot(X, y, 'sex', 'weight', ax=axes[3][0], ntrees=50,
                     alpha=.2,
                     min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    cat_partial_plot(X, y, 'pregnant', 'weight', ax=axes[4][0], ntrees=50,
                     alpha=.2,
                     min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'education', 'weight')
    plot_ICE(ice, 'education', 'weight', ax=axes[1, 1], yrange=(-12, 0))
    ice = ICE_predict(rf, X, 'height', 'weight')
    plot_ICE(ice, 'height', 'weight', ax=axes[2, 1], yrange=(0, 160))
    ice = ICE_predict(rf, X, 'sex', 'weight')
    plot_ICE(ice, 'sex', 'weight', ax=axes[3,1], yrange=(0,2), cats=df_raw['sex'].unique())
    ice = ICE_predict(rf, X, 'pregnant', 'weight')
    plot_ICE(ice, 'pregnant', 'weight', ax=axes[4,1], yrange=(0,10), cats=df_raw['pregnant'].unique())

    fig.suptitle("weight = 120 + 10*(height-min(height)) + 10*pregnant - 1.2*education", size=14)

    if False:
        # show importance as RF-piecewise linear plot see it
        rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, oob_score=True, n_jobs=-1)
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

    fig, axes = plt.subplots(4, 2, figsize=(8,8), gridspec_kw = {'height_ratios':[.2,3,3,3]})

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    partial_plot(X, y, 'dayofyear', 'temperature', ax=axes[1][0],
                 ntrees=50, min_samples_leaf=7)#, yrange=(-12,0))
    # partial_plot(X, y, 'education', 'weight', ntrees=20, min_samples_leaf=7, alpha=.2)
    cat_partial_plot(X, y, 'state', 'temperature', cats=catencoders['state'], ax=axes[2][0])#, yrange=(0,160))
    # cat_partial_plot(axes[2][0], X, y, 'sex', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    # cat_partial_plot(axes[3][0], X, y, 'pregnant', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'dayofyear', 'temperature')
    plot_ICE(ice, 'dayofyear', 'temperature', ax=axes[1, 1])  #, yrange=(-12,0))
    ice = ICE_predict(rf, X, 'state', 'temperature')
    plot_ICE(ice, 'state', 'temperature', cats=catencoders['state'], ax=axes[2, 1])  #, yrange=(-12,0))

    df = df_raw.copy()
    axes[3, 0].plot(df.loc[df['state'] == 'CA', 'dayofyear'],
             df.loc[df['state'] == 'CA', 'temperature'], label="CA")
    axes[3, 0].plot(df.loc[df['state'] == 'CO', 'dayofyear'],
             df.loc[df['state'] == 'CO', 'temperature'], label="CO")
    axes[3, 0].plot(df.loc[df['state'] == 'AZ', 'dayofyear'],
             df.loc[df['state'] == 'AZ', 'temperature'], label="AZ")
    axes[3, 0].plot(df.loc[df['state'] == 'WA', 'dayofyear'],
             df.loc[df['state'] == 'WA', 'temperature'], label="WA")
    axes[3, 0].legend()
    axes[3,0].set_title('Raw data')
    axes[3, 0].set_ylabel('Temperature')
    axes[3, 0].set_xlabel('Dataframe row index')

    plt.tight_layout()

    plt.savefig("/tmp/weather.svg")
    plt.show()

def interaction(f, n=50):
    df,eqn = f(n=n)

    X = df.drop('y', axis=1)
    y = df['y']
    min_samples_leaf = 2

    fig, axes = plt.subplots(4, 2, figsize=(10,13))

    axes[0,0].plot(range(len(df)), df['x1'], label="x1")
    axes[0,0].plot(range(len(df)), df['x2'], label="x2")
    axes[0,0].plot(range(len(df)), df['y'], label="y")
    axes[0, 0].set_xlabel("df row index")
    axes[0, 0].set_ylabel("df value")
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Raw data; {eqn}")

    # axes[0,1].get_xaxis().set_visible(False)
    # axes[0,1].axis('off')

    rtreeviz_univar(axes[0,1],
                    df['x1'], y,
                    feature_name='x1',
                    target_name='y',
                    min_samples_leaf=min_samples_leaf,
                    fontsize=10)
    axes[0,1].set_title(f'x1 space partition with min_samples_leaf={min_samples_leaf}')
    axes[0,1].set_xlabel("x1")
    axes[0,1].set_ylabel("y")

    print(df)
    print(f"x1 = {df['x1'].values.tolist()}")
    print(f"x2 = {df['x2'].values.tolist()}")
    print(f"y = {df['y'].values.tolist()}")
    axes[1,0].plot(df['x1'], y)
    axes[1,0].set_xlabel("x1")
    axes[1,0].set_ylabel("y")
    axes[1,1].plot(df['x2'], y)
    axes[1,1].set_xlabel("x2")
    axes[1,1].set_ylabel("y")

    partial_plot(X, y, 'x1', 'y', ax=axes[2][0],
                 ntrees=30, min_samples_leaf=min_samples_leaf)#, yrange=(0,40))
    # partial_plot(X, y, 'education', 'weight', ntrees=20, min_samples_leaf=7, alpha=.2)
    partial_plot(X, y, 'x2', 'y', ax=axes[3][0], min_samples_leaf=min_samples_leaf,
                 ntrees=30)#, yrange=(0,40))
    # cat_partial_plot(axes[2][0], X, y, 'sex', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['sex'].unique(), yrange=(0,2))
    # cat_partial_plot(axes[3][0], X, y, 'pregnant', 'weight', ntrees=50, min_samples_leaf=7, cats=df_raw['pregnant'].unique(), yrange=(0,10))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)

    ice = ICE_predict(rf, X, 'x1', 'y')
    plot_ICE(ice, 'x1', 'y', ax=axes[2, 1], yrange=(0, 40))
    ice = ICE_predict(rf, X, 'x2', 'y')
    plot_ICE(ice, 'x2', 'y', ax=axes[3, 1], yrange=(0, 40))

    plt.tight_layout()

    plt.savefig(f"/tmp/interaction-{n}.png")
    plt.show()


def wine():
    wine = load_wine()


def bigX():
    def bigX_data(n):
        x1 = np.random.uniform(-1, 1, size=n)
        x2 = np.random.uniform(-1, 1, size=n)
        x3 = np.random.uniform(-1, 1, size=n)

        y = 0.2 * x1 - 5 * x2 + 10 * x2 * np.where(x3 >= 0, 1, 0) + np.random.normal(0, 1, size=n)
        df = pd.DataFrame()
        df['x1'] = x1
        df['x2'] = x2
        df['x3'] = x3
        df['y'] = y
        return df

    n = 1000
    df = bigX_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(5, 2, figsize=(11, 14), gridspec_kw = {'height_ratios':[.1,4,4,4,4]})

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    axes[1,0].scatter(df['x3'], y, s=5, alpha=.7)
    axes[1,0].set_xlabel('x2')
    axes[1,0].set_ylabel('y')

    axes[1,1].scatter(df['x2'], df['y'], s=5, alpha=.7)
    axes[1,1].set_ylabel('y')
    axes[1,1].set_xlabel('x2')

    # Partial deriv is just 0.2 so this is correct. flat deriv curve, net effect line at slope .2
    # ICE is way too shallow and not line at n=1000 even
    partial_plot(X, y, 'x1', 'y', ax=axes[2,0])
    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Key is asking right question. Don't look at marginal plot and say obvious.
    # Ask what is net effect at every x2? 0.
    partial_plot(X, y, 'x2', 'y', ax=axes[3,0], yrange=(-4,4))
    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    partial_plot(X, y, 'x3', 'y', ax=axes[4,0], yrange=(-4,4))

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = ICE_predict(rf, X, 'x1', 'y', numx=10)
    plot_ICE(ice, 'x1', 'y', ax=axes[2, 1], yrange=(-.05,.5))

    ice = ICE_predict(rf, X, 'x2', 'y', numx=10)
    plot_ICE(ice, 'x2', 'y', ax=axes[3, 1])

    ice = ICE_predict(rf, X, 'x3', 'y', numx=10)
    plot_ICE(ice, 'x3', 'y', ax=axes[4, 1])

    fig.suptitle("$y = 0.2x_1 - 5x_2 + 10x_2\mathbb{1}_{x_3 \geq 0} + \epsilon$\n$x_1, x_2, x_3$ are U(-1,1)\nSample size "+str(n))
    plt.tight_layout()
    plt.show()


def boston():
    df = pd.read_csv('/Users/parrt/github/random-forest-importances/notebooks/data/boston.csv')
    X = df.drop('medv', axis=1)
    y = df['medv']

    """
    Wow. My net effect plots look kinda like the centered ICE c-ICE plots
    from paper: https://arxiv.org/pdf/1309.6392.pdf
    Mine are way smoother.
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), gridspec_kw = {'height_ratios':[.05,4,4]})

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    axes[1,0].scatter(df['age'], y, s=5, alpha=.7)
    axes[1,0].set_xlabel('age')
    axes[1,0].set_ylabel('median home value')

    partial_plot(X, y, 'age', 'medv', ax=axes[2,0], yrange=(-20,20))

    rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = ICE_predict(rf, X, 'age', 'medv', numx=10)
    plot_ICE(ice, 'age', 'medv', ax=axes[2, 1], yrange=(-20,20))

    fig.suptitle(f"Boston housing data {len(X)} training samples\nRandom Forest ntrees=100")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # cars()
    # rent()
    # weight()
    # weather()
    # interaction(crisscross=True)
    # interaction(n=200, crisscross=False)
    # bigX()
    boston()