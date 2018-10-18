import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from rfpimp import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_all = pd.read_csv("../notebooks/data/rent-cls.csv")

num_features = ['bathrooms','bedrooms','latitude','longitude','price']
target = 'interest_level'

df = df_all[num_features + [target]]

def test1():
    # compute median per num bedrooms
    df_median_price_per_bedrooms = df.groupby(by='bedrooms')['price'].median().reset_index()
    beds_to_median = df_median_price_per_bedrooms.to_dict(orient='dict')['price']
    df['median_price_per_bedrooms'] = df['bedrooms'].map(beds_to_median)
    # compute ratio of price to median price for that num of bedrooms
    df['price_to_median_beds'] = df['price'] / df['median_price_per_bedrooms']
    # ratio of num bedrooms to price
    df["beds_per_price"] = df["bedrooms"] / df["price"]
    # total rooms (bed, bath)
    df["beds_baths"] = df["bedrooms"]+df["bathrooms"]
    del df['median_price_per_bedrooms'] # don't need after computation

    df_train, df_test = train_test_split(df, test_size=0.15)

    X_train, y_train = df_train.drop('interest_level',axis=1), df_train['interest_level']
    X_test, y_test = df_test.drop('interest_level',axis=1), df_test['interest_level']

    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                max_features=1.0,
                                min_samples_leaf=10, oob_score=True)
    rf.fit(X_train, y_train)

    I = importances(rf, X_test, y_test)
    return I


def test2():
    df_train, df_test = train_test_split(df, test_size=0.15)
    X_train, y_train = df_train.drop('interest_level',axis=1), df_train['interest_level']
    X_test, y_test = df_test.drop('interest_level',axis=1), df_test['interest_level']
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                max_features=1.0,
                                min_samples_leaf=10, oob_score=True)
    rf.fit(X_train, y_train)
    I = importances(rf, X_test, y_test, features=['bedrooms','bathrooms',['latitude', 'longitude']])
    return I


def test3():

    cancer = load_breast_cancer()

    X, y = cancer.data, cancer.target
    # show first 5 columns only
    # df = pd.DataFrame(X[:, 0:10], columns=cancer.feature_names[0:10])
    df = pd.DataFrame(X, columns=cancer.feature_names)
    #df['diagnosis'] = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

    cl = RandomForestClassifier(n_estimators=20)
    cl.fit(X_train, y_train)

    I = importances(cl, X_test, y_test)
    return I


viz = plot_importances(test1())
viz.save(filename='/tmp/t.svg')
I = test2()
viz = plot_importances(I)
viz.save(filename='/tmp/t2.svg')

# I = test3()
# viz = plot_importances(I)
# viz.save(filename='/tmp/t3.svg')

#cancer = load_breast_cancer()
# X, y = cancer.data, cancer.target
# df = pd.DataFrame(X, columns=cancer.feature_names)
#viz = plot_dependence_heatmap(D, figsize=(12, 12))

# D = feature_dependence_matrix(df, n_samples=5000)
# viz = plot_dependence_heatmap(D, figsize=(4,4))
# viz.view()