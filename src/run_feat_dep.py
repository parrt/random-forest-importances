import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_object_dtype, is_categorical_dtype
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from timeit import default_timer as timer

from rfpimp import *

df = pd.read_feather("/Users/parrt/github/mlbook-private/data/bulldozer-train-num.feather")

X_train, y_train = df.drop('SalePrice', axis=1), df['SalePrice']

rf = RandomForestRegressor(n_estimators=50,
                           n_jobs=-1,
                           oob_score=True,
                           max_features=.4)

start = timer() # ------------

D = oob_dependences(rf, X_train, 2000) # like 10 seconds
DM = feature_dependence_matrix(rf, X_train, 2000) # like 15 minutes

end = timer() # ------------
print(f"{end - start:.2f}s")

print(D)
print(DM)
