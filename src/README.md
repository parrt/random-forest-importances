# Feature importances for scikit random forests

```
from rfimpo import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/Users/parrt/github/random-forest-importances/notebooks/data/rent.csv")
features = ['bathrooms','bedrooms','longitude','latitude',
            'price']
df = df[features].copy()
df['price'] = np.log(df['price'])
print(df.head(5))

rf = RandomForestRegressor(n_estimators=100,
                                min_samples_leaf=1,
                                n_jobs=-1,
                                oob_score=True)

X_train, y_train = df.drop('price',axis=1), df['price']
rf.fit(X_train, y_train)

imp = importances_df(rf, X_train, y_train)
plot_importances(X_train.columns, imp)

imp = dropcol_importances(rf, X_train, y_train)
plot_importances(X_train.columns, imp)
```
