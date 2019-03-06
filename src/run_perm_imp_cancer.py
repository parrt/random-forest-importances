import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from timeit import default_timer as timer

from rfpimp import *

df = pd.read_csv("/Users/parrt/github/qiforest/data/cancer.csv")
N = len(df)-20
target='diagnosis'
anomaly = df[df[target] == 1]
normal = df[df[target] == 0]
df = pd.concat([anomaly[0:20], normal[0:N]])

X, y = df.drop('diagnosis', axis=1), df['diagnosis']

weights = 1 / (np.bincount(y) / len(X))
rf = RandomForestClassifier(n_estimators=50,
                           n_jobs=-1,
                           oob_score=True,
                           class_weight={0: weights[0], 1: weights[1]},
                           max_features=.4)
rf.fit(X, y)
start = timer() # ------------

I = oob_importances(rf, X, y, n_samples=3000)
print(I)

# sample_weights = df.loc[df.target==0, ]
I = importances(rf, X, y, features=X.columns, n_samples=3000)
print(I)
end = timer() # ------------
print(f"{end - start:.2f}s")

viz = plot_importances(I)
viz.view()
