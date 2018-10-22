from rfpimp import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_orig = pd.read_csv("/Users/parrt/github/random-forest-importances/notebooks/data/rent.csv")

df = df_orig.copy()

# attentuate affect of outliers in price
df['price'] = np.log(df['price'])

df_train, df_test = train_test_split(df, test_size=0.20)

features = ['bathrooms','bedrooms','longitude','latitude',
            'price']
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('price',axis=1), df_train['price']
X_test, y_test = df_test.drop('price',axis=1), df_test['price']
X_train['random'] = np.random.random(size=len(X_train))
X_test['random'] = np.random.random(size=len(X_test))

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

imp = importances(rf, X_test, y_test) # permutation
viz = plot_importances(imp)
viz.view()


df_train, df_test = train_test_split(df_orig, test_size=0.20)
features = ['bathrooms','bedrooms','price','longitude','latitude',
            'interest_level']
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('interest_level',axis=1), df_train['interest_level']
X_test, y_test = df_test.drop('interest_level',axis=1), df_test['interest_level']
# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))
X_test['random'] = np.random.random(size=len(X_test))

rf = RandomForestClassifier(n_estimators=100,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            oob_score=True)
rf.fit(X_train, y_train)

imp = importances(rf, X_test, y_test, n_samples=-1)
viz = plot_importances(imp)
viz.view()

