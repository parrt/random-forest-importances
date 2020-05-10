"""
Attention! This is a slight customized version of rfpimp. The official
distribution had some functions broken by sklearn 0.22 version. As soon
as the official version gets fixed up, this file should be removed from
the project.


A simple library of functions that provide feature importances
for scikit-learn random forest regressors and classifiers.

MIT License
Terence Parr, http://parrt.cs.usfca.edu
Kerem Turgutlu, https://www.linkedin.com/in/kerem-turgutlu-12906b65
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.forest import _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.ensemble import forest
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from pandas.api.types import is_numeric_dtype
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from copy import copy
import warnings
import tempfile
from os import getpid, makedirs
from mpl_toolkits.axes_grid1 import make_axes_locatable

GREY = '#444443'


class PimpViz:
    """
    For use with jupyter notebooks, plot_importances returns an instance
    of this class so we display SVG not PNG.
    """
    def __init__(self):
        tmp = tempfile.gettempdir()
        self.svgfilename = tmp+"/PimpViz_"+str(getpid())+".svg"
        plt.tight_layout()
        plt.savefig(self.svgfilename, bbox_inches='tight', pad_inches=0)

    def _repr_svg_(self):
        with open(self.svgfilename, "r", encoding='UTF-8') as f:
            svg = f.read()
        plt.close()
        return svg

    def save(self, filename):
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def view(self):
        plt.show()

    def close(self):
        plt.close()


def importances(model, X_valid, y_valid, features=None, n_samples=5000, sort=True, metric=None, sample_weights = None):
    """
    Compute permutation feature importances for scikit-learn models using
    a validation set.

    Given a Classifier or Regressor in model
    and validation X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The validation data is needed to compute model performance
    measures (accuracy or R^2). The model is not retrained.

    You can pass in a list with a subset of features interesting to you.
    All unmentioned features will be grouped together into a single meta-feature
    on the graph. You can also pass in a list that has sublists like:
    [['latitude', 'longitude'], 'price', 'bedrooms']. Each string or sublist
    will be permuted together as a feature or meta-feature; the drop in
    overall accuracy of the model is the relative importance.

    The model.score() method is called to measure accuracy drops.

    This version that computes accuracy drops with the validation set
    is much faster than the OOB, cross validation, or drop column
    versions. The OOB version is a less vectorized because it needs to dig
    into the trees to get out of examples. The cross validation and drop column
    versions need to do retraining and are necessarily much slower.

    This function used OOB not validation sets in 1.0.5; switched to faster
    test set version for 1.0.6. (breaking API change)

    :param model: The scikit model fit to training data
    :param X_valid: Data frame with feature vectors of the validation set
    :param y_valid: Series with target variable of validation set
    :param features: The list of features to show in importance graph.
                     These can be strings (column names) or lists of column
                     names. E.g., features = ['bathrooms', ['latitude', 'longitude']].
                     Feature groups can overlap, with features appearing in multiple.
    :param n_samples: How many records of the validation set to use
                      to compute permutation importance. The default is
                      5000, which we arrived at by experiment over a few data sets.
                      As we cannot be sure how all data sets will react,
                      you can pass in whatever sample size you want. Pass in -1
                      to mean entire validation set. Our experiments show that
                      not too many records are needed to get an accurate picture of
                      feature importance.
    :param sort: Whether to sort the resulting importances
    :param metric: Metric in the form of callable(model, X_valid, y_valid, sample_weights) to evaluate for,
                    if not set default's to model.score()
    :param sample_weights: set if a different weighting is required for the validation samples


    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    X_train, y_train = ..., ...
    X_valid, y_valid = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_valid, y_valid)
    """
    def flatten(features):
        all_features = set()
        for sublist in features:
            if isinstance(sublist, str):
                all_features.add(sublist)
            else:
                for item in sublist:
                    all_features.add(item)
        return all_features

    if features is None:
        # each feature in its own group
        features = X_valid.columns.values
    else:
        req_feature_set = flatten(features)
        model_feature_set = set(X_valid.columns.values)
        # any features left over?
        other_feature_set = model_feature_set.difference(req_feature_set)
        if len(other_feature_set) > 0:
            # if leftovers, we need group together as single new feature
            features.append(list(other_feature_set))

    X_valid, y_valid, sample_weights = sample(X_valid, y_valid, n_samples, sample_weights=sample_weights)
    X_valid = X_valid.copy(deep=False)  # we're modifying columns

    if callable(metric):
        baseline = metric(model, X_valid, y_valid, sample_weights)
    else:
        baseline = model.score(X_valid, y_valid, sample_weights)

    imp = []
    for group in features:
        if isinstance(group, str):
            save = X_valid[group].copy()
            X_valid[group] = np.random.permutation(X_valid[group])
            if callable(metric):
                m = metric(model, X_valid, y_valid, sample_weights)
            else:
                m = model.score(X_valid, y_valid, sample_weights)
            X_valid[group] = save
        else:
            save = {}
            for col in group:
                save[col] = X_valid[col].copy()
            for col in group:
                X_valid[col] = np.random.permutation(X_valid[col])

            if callable(metric):
                m = metric(model, X_valid, y_valid, sample_weights)
            else:
                m = model.score(X_valid, y_valid, sample_weights)
            for col in group:
                X_valid[col] = save[col]
        imp.append(baseline - m)

    # Convert and groups/lists into string column names
    labels = []
    for col in features:
        if isinstance(col, list):
            labels.append('\n'.join(col))
        else:
            labels.append(col)

    I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(imp)})
    I = I.set_index('Feature')
    if sort:
        I = I.sort_values('Importance', ascending=False)
    return I


def sample(X_valid, y_valid, n_samples, sample_weights=None):
    if n_samples < 0: n_samples = len(X_valid)
    n_samples = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix = np.random.choice(len(X_valid), n_samples)
        X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid = y_valid.iloc[ix].copy(deep=False)
        if sample_weights is not None: sample_weights = sample_weights.iloc[ix].copy(deep=False)
    return X_valid, y_valid, sample_weights


def sample_rows(X, n_samples):
    if n_samples < 0: n_samples = len(X)
    n_samples = min(n_samples, len(X))
    if n_samples < len(X):
        ix = np.random.choice(len(X), n_samples)
        X = X.iloc[ix].copy(deep=False)  # shallow copy
    return X


def oob_importances(rf, X_train, y_train, n_samples=5000):
    """
    Compute permutation feature importances for scikit-learn
    RandomForestClassifier or RandomForestRegressor in arg rf.

    Given training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The training data is needed to compute out of bag (OOB)
    model performance measures (accuracy or R^2). The model
    is not retrained.

    By default, sample up to 5000 observations to compute feature importances.

    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = oob_importances(rf, X_train, y_train)
    """
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances(rf, X_train, y_train, oob_classifier_accuracy, n_samples)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances(rf, X_train, y_train, oob_regression_r2_score, n_samples)
    return None


def cv_importances(model, X_train, y_train, k=3):
    """
    Compute permutation feature importances for scikit-learn models using
    k-fold cross-validation (default k=3).

    Given a Classifier or Regressor in model
    and training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    Cross-validation observations are taken from X_train, y_train.

    The model.score() method is called to measure accuracy drops.

    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = cv_importances(rf, X_train, y_train)
    """
    def score(model):
        cvscore = cross_val_score(
            model,  # which model to use
            X_train, y_train,  # what training data to split up
            cv=k)  # number of folds/chunks
        return np.mean(cvscore)

    X_train = X_train.copy(deep=False)  # shallow copy
    baseline = score(model)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = score(model)
        X_train[col] = save
        imp.append(baseline - m)

    I = pd.DataFrame(data={'Feature': X_train.columns, 'Importance': np.array(imp)})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def permutation_importances(rf, X_train, y_train, metric, n_samples=5000):
    imp = permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def dropcol_importances(model, X_train, y_train, X_valid=None, y_valid=None, metric=None, sample_weights=None):
    """
    Compute drop-column feature importances for scikit-learn.

    Given a classifier or regression in model
    and training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.

    A clone of model is trained once to get the baseline score and then
    again, once per feature to compute the drop in either the model's .score() output
    or a custom metric callable in the form of metric(model, X_valid, y_valid).
    In case of a custom metric the X_valid and y_valid parameters should be set.

    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = dropcol_importances(rf, X_train, y_train)
    """
    if X_valid is None: X_valid = X_train
    if y_valid is None: y_valid = y_train
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(X_train, y_train)
    if callable(metric):
        baseline = metric(model_, X_valid, y_valid, sample_weights)
    else:
        baseline = model_.score(X_valid, y_valid, sample_weights)
    imp = []
    for col in X_train.columns:
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X_train.drop(col,axis=1), y_train)
        if callable(metric):
            s = metric(model_, X_valid.drop(col,axis=1), y_valid, sample_weights)
        else:
            s = model_.score(X_valid.drop(col,axis=1), y_valid, sample_weights)
        drop_in_score = baseline - s
        imp.append(drop_in_score)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def oob_dropcol_importances(rf, X_train, y_train):
    """
    Compute drop-column feature importances for scikit-learn.

    Given a RandomForestClassifier or RandomForestRegressor in rf
    and training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.

    A clone of rf is trained once to get the baseline score and then
    again, once per feature to compute the drop in out of bag (OOB)
    score.

    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = oob_dropcol_importances(rf, X_train, y_train)
    """
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.oob_score = True
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.oob_score = True
        rf_.fit(X_train.drop(col, axis=1), y_train)
        drop_in_score = baseline - rf_.oob_score_
        imp.append(drop_in_score)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def importances_raw(rf, X_train, y_train, n_samples=5000):
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances_raw(rf, X_train, y_train, oob_classifier_accuracy, n_samples)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances_raw(rf, X_train, y_train, oob_regression_r2_score, n_samples)
    return None


def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=5000):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    X_sample, y_sample, _ = sample(X_train, y_train, n_samples)

    if not hasattr(rf, 'estimators_'):
        rf.fit(X_sample, y_sample)

    baseline = metric(rf, X_sample, y_sample)
    X_train = X_sample.copy(deep=False) # shallow copy
    y_train = y_sample
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        drop_in_metric = baseline - m
        imp.append(drop_in_metric)
    return np.array(imp)


def _get_unsample_indices(tree, n_samples):
    """
    An interface to get unsampled indices regardless of sklearn version.
    """
    # Version 0.21 or older uses only two arguments.
    try:
        return _generate_unsampled_indices(tree.random_state, n_samples)
    # Version 0.22 or newer uses only two arguments.
    except TypeError:
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
        return _generate_unsampled_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )


def oob_classifier_accuracy(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:

    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    n_classes = len(np.unique(y))
    predictions = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_:
        unsampled_indices = _get_unsample_indices
        tree_preds = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds

    predicted_class_indexes = np.argmax(predictions, axis=1)
    predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]

    oob_score = np.mean(y == predicted_classes)
    return oob_score


def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:

    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _get_unsample_indices(tree, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score


def stemplot_importances(df_importances,
                         yrot=0,
                         label_fontsize=10,
                         width=4,
                         minheight=1.5,
                         vscale=1.0,
                         imp_range=(-.002, .15),
                         color='#375FA5',
                         bgcolor=None,  # seaborn uses '#F1F8FE'
                         xtick_precision=2,
                         title=None):
    GREY = '#444443'
    I = df_importances
    unit = 1

    imp = I.Importance.values
    mindrop = np.min(imp)
    maxdrop = np.max(imp)
    imp_padding = 0.002
    imp_range = (min(imp_range[0], mindrop - imp_padding), max(imp_range[1], maxdrop))

    barcounts = np.array([f.count('\n')+1 for f in I.index])
    N = np.sum(barcounts)
    ymax = N * unit
    # print(f"barcounts {barcounts}, N={N}, ymax={ymax}")
    height = max(minheight, ymax * .27 * vscale)

    plt.close()
    fig = plt.figure(figsize=(width,height))
    ax = plt.gca()
    ax.set_xlim(*imp_range)
    ax.set_ylim(0,ymax)
    ax.spines['top'].set_linewidth(.3)
    ax.spines['right'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    if bgcolor:
        ax.set_facecolor(bgcolor)

    yloc = []
    y = barcounts[0]*unit / 2
    yloc.append(y)
    for i in range(1,len(barcounts)):
        wprev = barcounts[i-1]
        w = barcounts[i]
        y += (wprev + w)/2 * unit
        yloc.append(y)
    yloc = np.array(yloc)
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
    ax.set_xticks([maxdrop, imp_range[1]])
    ax.tick_params(labelsize=label_fontsize, labelcolor=GREY)
    ax.invert_yaxis()  # labels read top-to-bottom
    if title:
        ax.set_title(title, fontsize=label_fontsize+1, fontname="Arial", color=GREY)

    plt.hlines(y=yloc, xmin=imp_range[0], xmax=imp, lw=barcounts*1.2, color=color)
    for i in range(len(I.index)):
        plt.plot(imp[i], yloc[i], "o", color=color, markersize=barcounts[i]+2)
    ax.set_yticks(yloc)
    ax.set_yticklabels(I.index, fontdict={'verticalalignment': 'center'})
    plt.tick_params(
        pad=0,
        axis='y',
        which='both',
        left=False)

    # rotate y-ticks
    if yrot is not None:
        plt.yticks(rotation=yrot)

    plt.tight_layout()

    return PimpViz()


def plot_importances(df_importances,
                     yrot=0,
                     label_fontsize=10,
                     width=4,
                     minheight=1.5,
                     vscale=1,
                     imp_range=(-.002, .15),
                     color='#D9E6F5',
                     bgcolor=None,  # seaborn uses '#F1F8FE'
                     xtick_precision=2,
                     title=None,
                     ax=None):
    """
    Given an array or data frame of importances, plot a horizontal bar chart
    showing the importance values.

    :param df_importances: A data frame with Feature, Importance columns
    :type df_importances: pd.DataFrame
    :param width: Figure width in default units (inches I think). Height determined
                  by number of features.
    :type width: int
    :param minheight: Minimum plot height in default matplotlib units (inches?)
    :type minheight: float
    :param vscale: Scale vertical plot (default .25) to make it taller
    :type vscale: float
    :param label_fontsize: Font size for feature names and importance values
    :type label_fontsize: int
    :param yrot: Degrees to rotate feature (Y axis) labels
    :type yrot: int
    :param label_fontsize:  The font size for the column names and x ticks
    :type label_fontsize:  int
    :param scalefig: Scale width and height of image (widthscale,heightscale)
    :type scalefig: 2-tuple of floats
    :param xtick_precision: How many digits after decimal for importance values.
    :type xtick_precision: int
    :param xtick_precision: Title of plot; set to None to avoid.
    :type xtick_precision: string
    :param ax: Matplotlib "axis" to plot into
    :return: None

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_test, y_test)
    viz = plot_importances(imp)
    viz.save('file.svg')
    viz.save('file.pdf')
    viz.view() # or just viz in notebook
    """
    I = df_importances
    unit = 1
    ypadding = .1

    imp = I.Importance.values
    mindrop = np.min(imp)
    maxdrop = np.max(imp)
    imp_padding = 0.002
    imp_range = (min(imp_range[0], mindrop - imp_padding), max(imp_range[1], maxdrop + imp_padding))

    barcounts = np.array([f.count('\n')+1 for f in I.index])
    N = np.sum(barcounts)
    ymax = N * unit + len(I.index) * ypadding + ypadding
    # print(f"barcounts {barcounts}, N={N}, ymax={ymax}")
    height = max(minheight, ymax * .2 * vscale)

    if ax is None:
        plt.close()
        fig, ax = plt.subplots(1,1,figsize=(width,height))
    ax.set_xlim(*imp_range)
    ax.set_ylim(0,ymax)
    ax.spines['top'].set_linewidth(.3)
    ax.spines['right'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    if bgcolor:
        ax.set_facecolor(bgcolor)

    yloc = []
    y = barcounts[0]*unit / 2 + ypadding
    yloc.append(y)
    for i in range(1,len(barcounts)):
        wprev = barcounts[i-1]
        w = barcounts[i]
        y += (wprev + w)/2 * unit + ypadding
        yloc.append(y)
    yloc = np.array(yloc)
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
    # too close to show both max and right edge?
    if maxdrop/imp_range[1] > 0.9 or maxdrop < 0.02:
        ax.set_xticks([0, imp_range[1]])
    else:
        ax.set_xticks([0, maxdrop, imp_range[1]])
    ax.tick_params(labelsize=label_fontsize, labelcolor=GREY)
    ax.invert_yaxis()  # labels read top-to-bottom
    if title:
        ax.set_title(title, fontsize=label_fontsize+1, fontname="Arial", color=GREY)

    barcontainer = ax.barh(y=yloc, width=imp,
                           height=barcounts*unit,
                           tick_label=I.index,
                           color=color, align='center')

    # Alter appearance of each bar
    for rect in barcontainer.patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(GREY)

    # rotate y-ticks
    if yrot is not None:
        ax.tick_params(labelrotation=yrot)

    return PimpViz()


def oob_dependences(rf, X_train, n_samples=5000):
    """
    Given a random forest model, rf, and training observation independent
    variables in X_train (a dataframe), compute the OOB R^2 score using each var
    as a dependent variable. We retrain rf for each var.    Only numeric columns are considered.

    By default, sample up to 5000 observations to compute feature dependencies.

    :return: Return a DataFrame with Feature/Dependence values for each variable. Feature is the dataframe index.
    """
    numcols = [col for col in X_train if is_numeric_dtype(X_train[col])]

    X_train = sample_rows(X_train, n_samples)

    df_dep = pd.DataFrame(columns=['Feature','Dependence'])
    df_dep = df_dep.set_index('Feature')
    for col in numcols:
        X, y = X_train.drop(col, axis=1), X_train[col]
        rf.fit(X, y)
        df_dep.loc[col] = rf.oob_score_
    df_dep = df_dep.sort_values('Dependence', ascending=False)
    return df_dep


def feature_dependence_matrix(X_train,
                              rfmodel=RandomForestRegressor(n_estimators=50, oob_score=True),
                              zero=0.001,
                              sort_by_dependence=False,
                              n_samples=5000):
    """
    Given training observation independent variables in X_train (a dataframe),
    compute the feature importance using each var as a dependent variable using
    a RandomForestRegressor (even if var is actually categorical).
    We retrain a random forest for each var as target using the others as
    independent vars.  Only numeric columns are considered.

    By default, sample up to 5000 observations to compute feature dependencies.

    If feature importance is less than zero arg, force to 0. Force all negatives to 0.0.
    Clip to 1.0 max. (Some importances could come back > 1.0 because removing that
    feature sends R^2 very negative.)

    :return: a non-symmetric data frame with the dependence matrix where each row is the importance of each var to the row's var used as a model target.
    """
    numeric_cols = [col for col in X_train if is_numeric_dtype(X_train[col])]

    X_train = sample_rows(X_train, n_samples)

    df_dep = pd.DataFrame(index=X_train.columns, columns=['Dependence']+X_train.columns.tolist())
    for i,col in enumerate(numeric_cols):
        X, y = X_train.drop(col, axis=1), X_train[col]
        rf = clone(rfmodel)
        rf.fit(X,y)
        imp = permutation_importances_raw(rf, X, y, oob_regression_r2_score, n_samples)
        """
        Some importances could come back > 1.0 because removing that feature sends R^2
        very negative. Clip them at 1.0.  Also, features with negative importance
        means that taking them out helps predict but we don't care about that here.
        We want to know which features are collinear/predictive. Clip at 0.0.
        """
        imp = np.clip(imp, a_min=0.0, a_max=1.0)
        imp[imp<zero] = 0.0
        imp = np.insert(imp, i, 1.0)
        df_dep.iloc[i] = np.insert(imp, 0, rf.oob_score_) # add overall dependence

    if sort_by_dependence:
        return df_dep.sort_values('Dependence', ascending=False)
    return df_dep


def plot_dependence_heatmap(D,
                            color_threshold=0.6,
                            threshold=0.03,
                            cmap=None,
                            figsize=None,
                            value_fontsize=8,
                            label_fontsize=9,
                            precision=2,
                            xrot=70,
                            grid=True):
    depdata = D.values.astype(float)

    ncols, nrows = depdata.shape
    if figsize:
        fig = plt.figure(figsize=figsize)
    colnames = list(D.columns.values)
    colnames[0] = "$\\bf "+colnames[0]+"$" # bold Dependence word
    plt.xticks(range(len(colnames)), colnames, rotation=xrot, horizontalalignment='right',
               fontsize=label_fontsize, color=GREY)
    plt.yticks(range(len(colnames[1:])), colnames[1:], verticalalignment='center',
               fontsize=label_fontsize, color=GREY)
    if cmap is None:
        cw = plt.get_cmap('coolwarm')
        cmap = ListedColormap([cw(x) for x in np.arange(color_threshold, .85, 0.01)])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cm = copy(cmap)
    cm.set_under(color='white')

    for x in range(ncols):
        for y in range(nrows):
            if (x+1) == y or depdata[x,y]<threshold:
                depdata[x,y] = 0

    if grid:
        plt.grid(True, which='major', alpha=.25)

    im = plt.imshow(depdata, cmap=cm, vmin=color_threshold, vmax=1.0, aspect='equal')
    cb = plt.colorbar(im,
                      fraction=0.046,
                      pad=0.04,
                      ticks=[color_threshold,color_threshold+(1-color_threshold)/2,1.0])
    cb.ax.tick_params(labelsize=label_fontsize, labelcolor=GREY, pad=0)
    cb.outline.set_edgecolor('white')

    plt.axvline(x=.5, lw=1, color=GREY)

    for x in range(ncols):
        for y in range(nrows):
            if (x+1) == y:
                plt.annotate('x', xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=value_fontsize, color=GREY)
            if (x+1) != y and not np.isclose(round(depdata[x, y],precision), 0.0):
                plt.annotate(myround(depdata[x, y], precision), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=value_fontsize, color=GREY)
    plt.tick_params(pad=0, axis='x', which='both')

    ax = plt.gca()
    ax.spines['top'].set_linewidth(.3)
    ax.spines['right'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_edgecolor(GREY)
    ax.spines['bottom'].set_linewidth(.3)

    plt.tight_layout()
    return PimpViz()


def feature_corr_matrix(df):
    """
    Return the Spearman's rank-order correlation between all pairs
    of features as a matrix with feature names as index and column names.
    The diagonal will be all 1.0 as features are self correlated.

    Spearman's correlation is the same thing as converting two variables
    to rank values and then running a standard Pearson's correlation
    on those ranked variables. Spearman's is nonparametric and does not
    assume a linear relationship between the variables; it looks for
    monotonic relationships.

    :param df_train: dataframe containing features as columns, and
                     without the target variable.
    :return: a data frame with the correlation matrix
    """
    corr = np.round(spearmanr(df).correlation, 4)
    df_corr = pd.DataFrame(data=corr, index=df.columns, columns=df.columns)
    return df_corr


def plot_corr_heatmap(df,
                      color_threshold=0.6,
                      cmap=None,
                      figsize=None,
                      value_fontsize=8,
                      label_fontsize=9,
                      precision=2,
                      xrot=80):
    """
    Display the feature spearman's correlation matrix as a heatmap with
    any abs(value)>color_threshold appearing with background color.

    Spearman's correlation is the same thing as converting two variables
    to rank values and then running a standard Pearson's correlation
    on those ranked variables. Spearman's is nonparametric and does not
    assume a linear relationship between the variables; it looks for
    monotonic relationships.

    SAMPLE CODE

    from rfpimp import plot_corr_heatmap
    viz = plot_corr_heatmap(df_train, save='/tmp/corrheatmap.svg',
                      figsize=(7,5), label_fontsize=13, value_fontsize=11)
    viz.view() # or just viz in notebook
    """
    corr = spearmanr(df).correlation
    if len(corr.shape) == 0:
        corr = np.array([[1.0, corr],
                         [corr, 1.0]])

    filtered = copy(corr)
    filtered = np.abs(filtered)  # work with abs but display negatives later
    mask = np.ones_like(corr)
    filtered[np.tril_indices_from(mask)] = -9999

    if cmap is None:
        cw = plt.get_cmap('coolwarm')
        cmap = ListedColormap([cw(x) for x in np.arange(color_threshold, .85, 0.01)])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cm = copy(cmap)
    cm.set_under(color='white')

    if figsize:
        plt.figure(figsize=figsize)
    im = plt.imshow(filtered, cmap=cm, vmin=color_threshold, vmax=1, aspect='equal')

    width, height = filtered.shape
    for x in range(width):
        for y in range(height):
            if x == y:
                plt.annotate('x', xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=value_fontsize, color=GREY)
            if x < y:
                plt.annotate(myround(corr[x, y], precision), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=value_fontsize, color=GREY)

    cb = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[color_threshold, color_threshold + (1 - color_threshold) / 2, 1.0])
    cb.ax.tick_params(labelsize=label_fontsize, labelcolor=GREY, )
    cb.outline.set_edgecolor('white')
    plt.xticks(range(width), df.columns, rotation=xrot, horizontalalignment='right',
               fontsize=label_fontsize, color=GREY)
    plt.yticks(range(width), df.columns, verticalalignment='center',
               fontsize=label_fontsize, color=GREY)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(.3)
    ax.spines['right'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)

    plt.tight_layout()
    return PimpViz()


def rfnnodes(rf):
    """Return the total number of decision and leaf nodes in all trees of the forest."""
    return sum(t.tree_.node_count for t in rf.estimators_)


def dectree_max_depth(tree):
    """
    Return the max depth of this tree in terms of how many nodes; a single
    root node gives height 1.
    """
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]): # decision node
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else:  # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


def rfmaxdepths(rf):
    """
    Return the max depth of all trees in rf forest in terms of how many nodes
    (a single root node for a single tree gives height 1)
    """
    return [dectree_max_depth(t.tree_) for t in rf.estimators_]


def jeremy_trick_RF_sample_size(n):
    # Jeremy's trick; hmm.. this won't work as a separate function?
    # def batch_size_for_node_splitting(rs, n_samples):
    #     forest.check_random_state(rs).randint(0, n_samples, 20000)
    # forest._generate_sample_indices = batch_size_for_node_splitting
    forest._generate_sample_indices = \
        (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))

def jeremy_trick_reset_RF_sample_size():
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def myround(v,ndigits=2):
    if np.isclose(v, 0.0):
        return "0"
    return format(v, '.' + str(ndigits) + 'f')