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
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import r2_score
import warnings

from sklearn.ensemble.forest import _generate_unsampled_indices


def importances(model, X_valid, y_valid, n_samples=3500, sort=True):
    """
    Compute permutation feature importances for scikit-learn models using
    a validation set.

    Given a Classifier or Regressor in model
    and validation X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The validation data is needed to compute model performance
    measures (accuracy or R^2). The model is not retrained.

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
    if n_samples<0: n_samples = len(X_valid)
    n_samples = min(n_samples, len(X_valid))
    if n_samples<len(X_valid):
        ix = np.random.choice(len(X_valid),n_samples)
        X_valid = X_valid.iloc[ix].copy(deep=False) # shallow copy
        y_valid = y_valid.iloc[ix].copy(deep=False)
    else:
        X_valid = X_valid.copy(deep=False) # we're modifying columns

    baseline = model.score(X_valid, y_valid)
    imp = []
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        m = model.score(X_valid, y_valid)
        X_valid[col] = save
        imp.append(baseline - m)

    I = pd.DataFrame(data={'Feature': X_valid.columns, 'Importance': np.array(imp)})
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
    :param figsize: Specify width and height of image (width,height)
    :type figsize: 2-tuple of floats
    :param scalefig: Scale width and height of image (widthscale,heightscale)
    :type scalefig: 2-tuple of floats
    :param showfig: Execute plt.show() if true (default is True). Sometimes
                    we want to draw multiple things before calling plt.show()
    :type showfig: bool
    :return: None

    SAMPLE CODE

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_train, y_train)
    plot_importances(imp)
    """
    I = df_importances

    fig = plt.figure()
    w, h = figsize if figsize else fig.get_size_inches()
    fig.set_size_inches(w*scalefig[0], h*scalefig[1], forward=True)
    ax = plt.gca()
    ax.barh(np.arange(len(I.index)), I.Importance, height=.6, tick_label=I.index)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

    # rotate x-ticks
    if xrot is not None:
        plt.xticks(rotation=xrot)

    # xticks freq
    xticks = ax.get_xticks()
    nticks = len(xticks)
    new_ticks = xticks[np.arange(0, nticks, step=tickstep)]
    ax.set_xticks(new_ticks)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show()
