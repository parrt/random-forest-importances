"""
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
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from matplotlib.colors import ListedColormap
from copy import copy
import warnings


def importances(model, X_valid, y_valid, features=None, n_samples=3500, sort=True):
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
                      3500, which we arrived at by experiment over a few data sets.
                      As we cannot be sure how all data sets will react,
                      you can pass in whatever sample size you want. Pass in -1
                      to mean entire validation set. Our experiments show that
                      not too many records are needed to get an accurate picture of
                      feature importance.
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

    if not features:
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

    if n_samples < 0: n_samples = len(X_valid)
    n_samples = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix = np.random.choice(len(X_valid), n_samples)
        X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid = y_valid.iloc[ix].copy(deep=False)
    else:
        X_valid = X_valid.copy(deep=False)  # we're modifying columns

    baseline = model.score(X_valid, y_valid)
    imp = []
    for group in features:
        if isinstance(group, str):
            save = X_valid[group].copy()
            X_valid[group] = np.random.permutation(X_valid[group])
            m = model.score(X_valid, y_valid)
            X_valid[group] = save
        else:
            save = {}
            for col in group:
                save[col] = X_valid[col].copy()
            for col in group:
                X_valid[col] = np.random.permutation(X_valid[col])
            m = model.score(X_valid, y_valid)
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
        I = I.sort_values('Importance', ascending=True)
    return I


def oob_importances(rf, X_train, y_train):
    """
    Compute permutation feature importances for scikit-learn
    RandomForestClassifier or RandomForestRegressor in arg rf.

    Given training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The training data is needed to compute out of bag (OOB)
    model performance measures (accuracy or R^2). The model
    is not retrained.

    return: A data frame with Feature, Importance columns

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = oob_importances(rf, X_train, y_train)
    """
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances(rf, X_train, y_train, oob_classifier_accuracy)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances(rf, X_train, y_train, oob_regression_r2_score)
    return None


def cv_importances(model, X_train, y_train, k=3):
    """
    Compute permutation feature importances for scikit-learn models using
    k-fold cross-validation (default k=3).

    Given a Classifier or Regressor in model
    and training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The validation data is needed to compute model performance
    measures (accuracy or R^2). The model is retrained during
    cross-validation like with dropcol_dropcol_importances().

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
    I = I.sort_values('Importance', ascending=True)
    return I


def permutation_importances(rf, X_train, y_train, metric):
    imp = permutation_importances_raw(rf, X_train, y_train, metric)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I


def dropcol_importances(rf, X_train, y_train):
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
    imp = dropcol_importances(rf, X_train, y_train)
    """
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I


def importances_raw(rf, X_train, y_train):
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances_raw(rf, X_train, y_train, oob_classifier_accuracy)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances_raw(rf, X_train, y_train, oob_regression_r2_score)
    return None


def permutation_importances_raw(rf, X_train, y_train, metric):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    baseline = metric(rf, X_train, y_train)
    X_train = X_train.copy(deep=False) # shallow copy
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)


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
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
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
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score


def plot_importances(df_importances, save=None, xrot=0, tickstep=3,
                     label_fontsize=12,
                     figsize=None, scalefig=(1.0, 1.0), show=True):
    """
    Given an array or data frame of importances, plot a horizontal bar chart
    showing the importance values.

    :param df_importances: A data frame with Feature, Importance columns
    :type df_importances: pd.DataFrame
    :param save: A filename identifying where to save the image.
    :param xrot: Degrees to rotate importance (X axis) labels
    :type xrot: int
    :param tickstep: How many ticks to skip in X axis
    :type tickstep: int
    :param label_fontsize:  The font size for the column names and x ticks
    :type label_fontsize:  int
    :param figsize: Specify width and height of image (width,height)
    :type figsize: 2-tuple of floats
    :param scalefig: Scale width and height of image (widthscale,heightscale)
    :type scalefig: 2-tuple of floats
    :param show: Execute plt.show() if true (default is True). Sometimes
                 we want to draw multiple things before calling plt.show()
    :type show: bool
    :return: None

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_test, y_test)
    plot_importances(imp)
    """
    I = df_importances

    if figsize:
        fig = plt.figure(figsize=figsize)
    elif scalefig:
        fig = plt.figure()
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scalefig[0], h * scalefig[1], forward=True)
    else:
        fig = plt.figure()
    ax = plt.gca()
    labels = []
    for col in I.index:
        if isinstance(col, list):
            labels.append('\n'.join(col))
        else:
            labels.append(col)

    for tick in ax.get_xticklabels():
        tick.set_size(label_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_size(label_fontsize)
    ax.barh(np.arange(len(I.index)), I.Importance, height=0.6, tick_label=labels)

    # rotate x-ticks
    if xrot is not None:
        plt.xticks(rotation=xrot)

    # xticks freq
    xticks = ax.get_xticks()
    nticks = len(xticks)
    new_ticks = xticks[np.arange(0, nticks, step=tickstep)]
    ax.set_xticks(new_ticks)

    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show()


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
                      threshold=0.6,
                      cmap=None,
                      figsize=None,
                      value_fontsize=12, label_fontsize=14,
                      xrot=80,
                      save=None,
                      show=True):
    """
    Display the feature spearman's correlation matrix as a heatmap with
    any abs(value)>threshold appearing with background color.

    Spearman's correlation is the same thing as converting two variables
    to rank values and then running a standard Pearson's correlation
    on those ranked variables. Spearman's is nonparametric and does not
    assume a linear relationship between the variables; it looks for
    monotonic relationships.

    SAMPLE CODE

    from rfpimp import plot_corr_heatmap
    plot_corr_heatmap(df_train, save='/tmp/corrheatmap.svg',
                      figsize=(7,5), label_fontsize=13, value_fontsize=11)

    """
    corr = np.round(spearmanr(df).correlation, 4)

    filtered = copy(corr)
    filtered = np.abs(filtered)  # work with abs but display negatives later
    mask = np.ones_like(corr)
    filtered[np.tril_indices_from(mask)] = -9999

    if not cmap:
        cw = plt.get_cmap('coolwarm')
        cmap = ListedColormap([cw(x) for x in np.arange(.6, .85, 0.01)])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cm = copy(cmap)
    cm.set_under(color='white')

    if figsize:
        plt.figure(figsize=figsize)
    plt.imshow(filtered, cmap=cm, vmin=threshold, vmax=1, aspect='equal')

    width, height = filtered.shape
    for x in range(width):
        for y in range(height):
            if x < y:
                plt.annotate(str(np.round(corr[x, y], 2)), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=value_fontsize)
    plt.colorbar()
    plt.xticks(range(width), df.columns, rotation=xrot, horizontalalignment='right',
               fontsize=label_fontsize)
    plt.yticks(range(width), df.columns, verticalalignment='center',
               fontsize=label_fontsize)

    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show()
