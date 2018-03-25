import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score
import warnings

from sklearn.ensemble.forest import _generate_unsampled_indices


def importances(rf, X_train, y_train):
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances(rf, X_train, y_train, oob_classifier_accuracy)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances(rf, X_train, y_train, oob_regression_r2_score)
    return None


def importances_df(rf, X_train, y_train):
    if isinstance(rf, RandomForestClassifier):
        return permutation_importances_df(rf, X_train, y_train, oob_classifier_accuracy)
    elif isinstance(rf, RandomForestRegressor):
        return permutation_importances_df(rf, X_train, y_train, oob_regression_r2_score)
    return None


def permutation_importances(rf, X_train, y_train, metric):
    """
    Return importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def permutation_importances_df(rf, X_train, y_train, metric):
    imp = permutation_importances(rf, X_train, y_train, metric)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I


def dropcol_importances(rf, X_train, y_train):
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
    return np.array(imp)


def oob_classifier_accuracy(rf, X_train, y_train):
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
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    n_classes = len(np.unique(y))
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


def plot_importances(columns, importances,figsize=None, save=None, xrot=None, tickstep=2):
    I = importances
    if not isinstance(I,pd.DataFrame):
        I = pd.DataFrame(data={'Feature':columns, 'Importance':importances})
        I = I.set_index('Feature')
        I = I.sort_values('Importance', ascending=True)
    
    I.plot(kind='barh', figsize=figsize, legend=False, fontsize=14)

    ax = plt.gca()

    # remove y label
    y_axis = ax.axes.get_yaxis()
    y_label = y_axis.get_label()
    y_label.set_visible(False)
    
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
        plt.savefig(save)
    plt.show()
