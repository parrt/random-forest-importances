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

rf = RandomForestRegressor(n_estimators=50,
                           n_jobs=-1,
                           oob_score=True,
                           max_features=.4)
X_train, y_train = df.drop('SalePrice', axis=1), df['SalePrice']

start = timer() # ------------

I = oob_importances(rf, X_train, y_train, n_samples=3000)

end = timer() # ------------
print(f"{end - start:.2f}s")

viz = plot_importances(I)
viz.view()
