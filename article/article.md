# Default Random Forest Importances Are Inaccurate

<pyeval label=rent output="df.head(3) # dump first 3 rows">
import pandas as pd
df = pd.read_csv("data/rent.csv")
</pyeval>

<pyeval label=rent>
from sklearn.ensemble import RandomForestClassifier

X_train, y_train = df.drop('interest_level',axis=1), df['interest_level']
rf = RandomForestClassifier(n_estimators=100,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            oob_score=True)
rf.fit(X_train, y_train)
print(f"RF OOB accuracy {rf.oob_score_:.4f}")
</pyeval>

foo[cls_gini]

<figure label="cls_gini" caption="foo">
<pyfig label=rent hide=true width="35%">
import matplotlib.pyplot as plt

def plot_importances(columns,importances,figsize=None):
    I = pd.DataFrame(data={'Feature':columns, 'Importance':importances})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    I.plot(kind='barh', figsize=figsize, legend=False, fontsize=16)
	plt.tight_layout()
	plt.show()
	
plot_importances(X_train.columns,rf.feature_importances_)
</pyfig>
</figure>

<figure label="cls_gini" caption="foo">
<pyfig label=rent hide=true width="35%">
import numpy as np
from sklearn.base import clone

X_train2 = X_train.copy()
X_train2['random'] = np.random.random(size=len(X_train2))
rf2 = clone(rf)
rf2.fit(X_train2, y_train)
plot_importances(X_train2.columns,rf2.feature_importances_)
</pyfig>
</figure>

<pyeval label=rent output="dfcls.head(2)">
dfcls = df.copy()
dfcls['price'] = np.log(dfcls['price'])
X_train, y_train = dfcls.drop(['price','interest_level'],axis=1), dfcls['price']
</pyeval>

<pyfig label=rent side=true>
from sklearn.ensemble import RandomForestRegressor

rfcls = RandomForestRegressor(n_estimators=100,
	                          min_samples_leaf=1,
	                          n_jobs=-1,
	                          oob_score=True)
rfcls.fit(X_train, y_train)
plot_importances(X_train.columns, rfcls.feature_importances_)
</pyfig>

<pyfig label=rent side=true>
X_train2 = X_train.copy()
X_train2['random'] = np.random.random(size=len(X_train2))
rfcls2 = clone(rfcls)
rfcls2.fit(X_train2, y_train)
plot_importances(X_train2.columns,rfcls2.feature_importances_)
</pyfig>
