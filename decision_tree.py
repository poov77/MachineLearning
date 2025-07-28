# Converted from decision_tree.ipynb

# Code Cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code Cell
dataframe=pd.read_csv("WineQT.csv")

# Code Cell
dataframe.head()

# Code Cell
dataframe.columns

# Code Cell
set(dataframe["quality"])

# Code Cell
#EDA

# Code Cell
dataframe.corr()

# Code Cell
dataframe.describe()

# Code Cell
plt.figure(figsize=(13,8))
sns.heatmap(dataframe.corr(),annot=True,cmap="coolwarm")
plt.title("matrix view")
plt.show()

# Code Cell
dataframe.shape
dataframe.columns
dataframe.info()
dataframe.describe()


# Code Cell
#u=check missing values

dataframe.isnull().sum()


# Code Cell
#Correaltion Analysis

sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')


# Code Cell
#Distribution PLots

sns.histplot()
kde=True

# Code Cell
#boxplot for outliers

for col in dataframe.columns:
    if dataframe[col].dtype != 'object':
        sns.boxplot(x=dataframe[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


# Code Cell
#categories vs Target

sns.countplot(x='quality', data=dataframe)
sns.boxplot(x='quality', y='alcohol', data=dataframe)



# Code Cell
#Pairplot to inspect feature interactions:

sns.pairplot(dataframe[['alcohol', 'quality', 'sulphates', 'volatile acidity']])


# Code Cell
#missing value checking

print(dataframe.columns.tolist())


# Code Cell
dataframe.columns = dataframe.columns.str.strip()


# Code Cell
dataframe.columns

# Code Cell
dataframe[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality', 'Id']]

# Code Cell
#feature realtionship

sns.heatmap(dataframe.corr(), annot=True)
sns.boxplot(x='pH', y='quality', data=dataframe)


# Code Cell
import numpy as np

# Flag 0s as NaN in key columns
suspect_cols = ['fixed acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']

dataframe[suspect_cols] = dataframe[suspect_cols].replace(0, np.nan)

# View missing count
print(dataframe[suspect_cols].isnull().sum())


# Code Cell
import seaborn as sns
import matplotlib.pyplot as plt

for col in suspect_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(dataframe[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# Code Cell
sns.pairplot(dataframe[['alcohol', 'quality', 'sulphates', 'volatile acidity']],
             hue='quality', palette='husl')
plt.suptitle("Pairwise Relationships Among Key Features", y=1.02)


# Code Cell
sns.countplot(x='quality', data=dataframe, palette='Set2')
plt.title("Wine Quality Distribution")
plt.show()


# Code Cell
#splitting into x and Y

# Code Cell
x=dataframe.drop(columns='quality',axis=1)
y=dataframe['quality']

# Code Cell
x.columns


# Code Cell
#training and testing

# Code Cell
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y)


# Code Cell
#data modeling

# Code Cell
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train, y_train)

# Code Cell
#model prediction

# Code Cell
y_pred=model.predict(x_test)
y_pred

# Code Cell
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(model,filled=True)

# Code Cell
# reducing data size

sampleData=dataframe.head(20)
sampleData

# Code Cell
x_sample=sampleData.drop(columns='quality',axis=1)
y_sample=sampleData['quality']

# Code Cell
sampleModel=DecisionTreeClassifier()
sampleModel.fit(x_sample,y_sample)

# Code Cell
sampleData.columns

# Code Cell
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(sampleModel,filled=True)

# Code Cell
#reason of three values 

set(sampleData.quality)

# Code Cell
#model evoluation

# Code Cell
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

# Code Cell
#Random forest classifier


# Code Cell
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

# Code Cell
rf.score(x_test,y_test)

# Code Cell
y_predict_rf=rf.predict(x_test)
print(y_predict_rf)

# Code Cell
rf.estimators_

# Code Cell
plt.figure(figsize=(10,10))
tree.plot_tree(rf.estimators_[0],filled=True)

# Code Cell
lst=[DecisionTreeClassifier(max_features='sqrt', random_state=234495781),
 DecisionTreeClassifier(max_features='sqrt', random_state=1315527214),
 DecisionTreeClassifier(max_features='sqrt', random_state=2065493311),
 DecisionTreeClassifier(max_features='sqrt', random_state=518839515),
 DecisionTreeClassifier(max_features='sqrt', random_state=574172618),
 DecisionTreeClassifier(max_features='sqrt', random_state=1193573467),
 DecisionTreeClassifier(max_features='sqrt', random_state=1547755361),
 DecisionTreeClassifier(max_features='sqrt', random_state=1561627490),
 DecisionTreeClassifier(max_features='sqrt', random_state=617413769),
 DecisionTreeClassifier(max_features='sqrt', random_state=142329105),
 DecisionTreeClassifier(max_features='sqrt', random_state=804892943),
 DecisionTreeClassifier(max_features='sqrt', random_state=1842931465),
 DecisionTreeClassifier(max_features='sqrt', random_state=2119749969),
 DecisionTreeClassifier(max_features='sqrt', random_state=254127678),
 DecisionTreeClassifier(max_features='sqrt', random_state=144493324),
 DecisionTreeClassifier(max_features='sqrt', random_state=447906888),
 DecisionTreeClassifier(max_features='sqrt', random_state=1111262619),
 DecisionTreeClassifier(max_features='sqrt', random_state=1784961376),
 DecisionTreeClassifier(max_features='sqrt', random_state=1380678333),
 DecisionTreeClassifier(max_features='sqrt', random_state=471636108),
 DecisionTreeClassifier(max_features='sqrt', random_state=109435516),
 DecisionTreeClassifier(max_features='sqrt', random_state=1121991259),
 DecisionTreeClassifier(max_features='sqrt', random_state=994886649),
 DecisionTreeClassifier(max_features='sqrt', random_state=1388226063),
 DecisionTreeClassifier(max_features='sqrt', random_state=2101002591),
 DecisionTreeClassifier(max_features='sqrt', random_state=79906296),
 DecisionTreeClassifier(max_features='sqrt', random_state=1085855522),
 DecisionTreeClassifier(max_features='sqrt', random_state=339671473),
 DecisionTreeClassifier(max_features='sqrt', random_state=1379863843),
 DecisionTreeClassifier(max_features='sqrt', random_state=616427054),
 DecisionTreeClassifier(max_features='sqrt', random_state=609459996),
 DecisionTreeClassifier(max_features='sqrt', random_state=969049706),
 DecisionTreeClassifier(max_features='sqrt', random_state=185523677),
 DecisionTreeClassifier(max_features='sqrt', random_state=516275234),
 DecisionTreeClassifier(max_features='sqrt', random_state=437204668),
 DecisionTreeClassifier(max_features='sqrt', random_state=80827513),
 DecisionTreeClassifier(max_features='sqrt', random_state=736978323),
 DecisionTreeClassifier(max_features='sqrt', random_state=643333271),
 DecisionTreeClassifier(max_features='sqrt', random_state=1069306105),
 DecisionTreeClassifier(max_features='sqrt', random_state=1550562956),
 DecisionTreeClassifier(max_features='sqrt', random_state=1236308392),
 DecisionTreeClassifier(max_features='sqrt', random_state=1505100828),
 DecisionTreeClassifier(max_features='sqrt', random_state=1458479265),
 DecisionTreeClassifier(max_features='sqrt', random_state=223673902),
 DecisionTreeClassifier(max_features='sqrt', random_state=2063857725),
 DecisionTreeClassifier(max_features='sqrt', random_state=187627091),
 DecisionTreeClassifier(max_features='sqrt', random_state=502981292),
 DecisionTreeClassifier(max_features='sqrt', random_state=753334247),
 DecisionTreeClassifier(max_features='sqrt', random_state=1709656856),
 DecisionTreeClassifier(max_features='sqrt', random_state=1857301175),
 DecisionTreeClassifier(max_features='sqrt', random_state=2042838618),
 DecisionTreeClassifier(max_features='sqrt', random_state=2086594277),
 DecisionTreeClassifier(max_features='sqrt', random_state=570767012),
 DecisionTreeClassifier(max_features='sqrt', random_state=557370185),
 DecisionTreeClassifier(max_features='sqrt', random_state=1460407713),
 DecisionTreeClassifier(max_features='sqrt', random_state=1880692327),
 DecisionTreeClassifier(max_features='sqrt', random_state=1400689713),
 DecisionTreeClassifier(max_features='sqrt', random_state=1028429188),
 DecisionTreeClassifier(max_features='sqrt', random_state=1881174615),
 DecisionTreeClassifier(max_features='sqrt', random_state=636993610),
 DecisionTreeClassifier(max_features='sqrt', random_state=2049790056),
 DecisionTreeClassifier(max_features='sqrt', random_state=991635502),
 DecisionTreeClassifier(max_features='sqrt', random_state=691815677),
 DecisionTreeClassifier(max_features='sqrt', random_state=911179973),
 DecisionTreeClassifier(max_features='sqrt', random_state=394130173),
 DecisionTreeClassifier(max_features='sqrt', random_state=1594088442),
 DecisionTreeClassifier(max_features='sqrt', random_state=1401998687),
 DecisionTreeClassifier(max_features='sqrt', random_state=894269349),
 DecisionTreeClassifier(max_features='sqrt', random_state=712592422),
 DecisionTreeClassifier(max_features='sqrt', random_state=1486384945),
 DecisionTreeClassifier(max_features='sqrt', random_state=260655411),
 DecisionTreeClassifier(max_features='sqrt', random_state=1327359988),
 DecisionTreeClassifier(max_features='sqrt', random_state=747039640),
 DecisionTreeClassifier(max_features='sqrt', random_state=1902763471),
 DecisionTreeClassifier(max_features='sqrt', random_state=513241775),
 DecisionTreeClassifier(max_features='sqrt', random_state=1844636598),
 DecisionTreeClassifier(max_features='sqrt', random_state=704401553),
 DecisionTreeClassifier(max_features='sqrt', random_state=1465272016),
 DecisionTreeClassifier(max_features='sqrt', random_state=2022038624),
 DecisionTreeClassifier(max_features='sqrt', random_state=1362166342),
 DecisionTreeClassifier(max_features='sqrt', random_state=1978600050),
 DecisionTreeClassifier(max_features='sqrt', random_state=1905205852),
 DecisionTreeClassifier(max_features='sqrt', random_state=244430675),
 DecisionTreeClassifier(max_features='sqrt', random_state=1328407167),
 DecisionTreeClassifier(max_features='sqrt', random_state=1160548429),
 DecisionTreeClassifier(max_features='sqrt', random_state=1411197528),
 DecisionTreeClassifier(max_features='sqrt', random_state=1533241838),
 DecisionTreeClassifier(max_features='sqrt', random_state=1709624254),
 DecisionTreeClassifier(max_features='sqrt', random_state=615974016),
 DecisionTreeClassifier(max_features='sqrt', random_state=758128),
 DecisionTreeClassifier(max_features='sqrt', random_state=1143520286),
 DecisionTreeClassifier(max_features='sqrt', random_state=1012277934),
 DecisionTreeClassifier(max_features='sqrt', random_state=1403997446),
 DecisionTreeClassifier(max_features='sqrt', random_state=1716373182),
 DecisionTreeClassifier(max_features='sqrt', random_state=225887404),
 DecisionTreeClassifier(max_features='sqrt', random_state=1820479719),
 DecisionTreeClassifier(max_features='sqrt', random_state=1667656174),
 DecisionTreeClassifier(max_features='sqrt', random_state=198896365),
 DecisionTreeClassifier(max_features='sqrt', random_state=1202949791),
 DecisionTreeClassifier(max_features='sqrt', random_state=1984333749)]

# Code Cell
print(len(lst))

# Code Cell
grid_param = {
    "n_estimators": [5, 10, 50, 100, 120, 150],
    "criterion": ["gini", "entropy"],
    "max_depth": range(5),  # range(10) gives [0–9], so I adjusted to [1–10]
    # "min_samples_leaf": list(range(1, 11))  # range(10) also starts from 0 by default
}


# Code Cell
from sklearn.model_selection import GridSearchCV

grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=grid_param,  # Make sure this matches your earlier dictionary name
    cv=10,
    n_jobs=-6,              # Should be negative only if you want to use all CPUs; use n_jobs=-1 for max cores
    verbose=1
)

grid_search_rf.fit(x_train, y_train)


# Code Cell
grid_search_rf.best_params_

# Code Cell
rf_new=RandomForestClassifier(n_estimators=120, criterion='entropy',max_depth=4)
rf_new.fit(x_train, y_train)

# Code Cell
rf_new.score(x_train,y_train)

# Code Cell
#explore the implementation of randomsearchCV on the top of randomforest algorithm

# Code Cell
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 1. Define the model
rf = RandomForestClassifier(random_state=42)

# 2. Define hyperparameter distributions
param_dist = {
    "n_estimators": randint(10,100 ),
    "max_depth": randint(2, 10),
    "min_samples_split": randint(2, 100),
    "min_samples_leaf": randint(1, 10),
    "criterion": ["gini", "entropy"]
}

# 3. Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,             # Number of parameter settings sampled
    cv=10,                 # 10-fold cross-validation
    n_jobs=-1,             # Use all CPU cores
    verbose=1,
    random_state=42
)

# 4. Fit to data
random_search.fit(x_train, y_train)


# Code Cell
# Best parameters
print("Best Parameters:", random_search.best_params_)

# Best estimator
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(x_test)

# Performance report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Code Cell
plt.close()
