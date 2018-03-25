# Feature importances for scikit random forests

By <a href="http://parrt.cs.usfca.edu">Terence Parr</a> and <a href="https://www.linkedin.com/in/kerem-turgutlu-12906b65/">Kerem Turgutlu</a>.

The scikit-learn Random Forest feature importances strategy is <i>mean decrease in impurity</i> (or <i>gini importance</i>) mechanism, which is unreliable.
To get reliable results, use permutation importance, provided in this `rfpimp` package.

## Description

The mean-decrease-in-impurity importance of a feature is computed by measuring how effective the feature is at reducing uncertainty (classifiers) or variance (regressors) when creating decision trees within random forests.  The problem is that this mechanism, while fast, does not always give an accurate picture of importance. Strobl <i>et al</i> pointed out in <a href="https://link.springer.com/article/10.1186%2F1471-2105-8-25">Bias in random forest variable importance measures: Illustrations, sources and a solution</a> that &ldquo;<i>the variable importance measures of Breiman's original random forest method ... are not reliable in situations where potential predictor variables vary in their scale of measurement or their number of categories</i>.&rdquo; 

A more reliable method is <i>permutation importance</i>, which measures the importance of a feature as follows. Record a baseline accuracy (classifier) or R<sup>2</sup> score (regressor) by passing a  validation set or the out-of-bag (OOB) samples through the random forest.  Permute the column values of a single predictor feature and then pass all test samples back through the random forest and recompute the accuracy or R<sup>2</sup>. The importance of that feature is the difference between the baseline and the drop in overall accuracy or R<sup>2</sup> caused by permuting the column. The permutation mechanism is much more computationally expensive than the mean decrease in impurity mechanism, but the results are more reliable.

importances_raw
importances
permutation_importances_raw
permutation_importances
dropcol_importances
oob_classifier_accuracy
oob_regression_r2_score
plot_importances

## Sample code

```python
from rfpimp import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/Users/parrt/github/random-forest-importances/notebooks/data/rent.csv")

#df = df.iloc[0:5000]

print(df.head(5))

# Regressor

features = ['bathrooms','bedrooms','longitude','latitude',
            'price']
dfr = df[features].copy()

dfr['price'] = np.log(dfr['price'])

rf = RandomForestRegressor(n_estimators=100,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           oob_score=True)

X_train, y_train = dfr.drop('price',axis=1), dfr['price']
# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))
rf.fit(X_train, y_train)

imp = importances(rf, X_train, y_train) # permutation
plot_importances(imp)

imp = dropcol_importances(rf, X_train, y_train)
plot_importances(imp)

# Classifier

features = ['bathrooms','bedrooms','price','longitude','latitude',
            'interest_level']
dfc = df[features].copy()

X_train, y_train = dfc.drop('interest_level',axis=1), dfc['interest_level']
# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))

rf = RandomForestClassifier(n_estimators=100,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            oob_score=True)
rf.fit(X_train, y_train)

imp = importances(rf, X_train, y_train) # permutation
plot_importances(imp)

rf = RandomForestClassifier(n_estimators=100,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            oob_score=True)
imp = dropcol_importances(rf, X_train, y_train)
plot_importances(imp)
```
