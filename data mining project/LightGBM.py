import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree, preprocessing

import lightgbm as lgb

lgb_estimator=lgb.LGBMClassifier()

train_data = pd.read_csv("train.csv")
y=train_data[["prognosis"]]
X=train_data.drop(columns=["prognosis","id"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

## tuning
# parameters={
#         'num_leaves': range(10,15,1),
#     'n_estimators': range(50,70,1),
#     'min_child_samples' : range(3,5,1),
#     'max_depth' : range(6,8,1),
#     'learning_rate': [x/10 for x in range(1,5,1)]
#
# }
# print("X_train.shape:", X_train.shape)
# grid_search_estimator = GridSearchCV(lgb_estimator, parameters, scoring='accuracy', cv=10)
# grid_search_estimator.fit(X_train, y_train)
# result=grid_search_estimator.cv_results_
# # print the results of all hyper-parameter combinations
# results = pd.DataFrame(grid_search_estimator.cv_results_)
# xg_bestScore=grid_search_estimator.best_score_
# xg_bestParams=grid_search_estimator.best_params_
# print the best parameter setting
# print("best score is {} with params {}".format(xg_bestScore, xg_bestParams))

## apply model
test=pd.read_csv("test.csv").drop(columns=["id"])
estimator=lgb.LGBMClassifier(learning_rate=0.1, max_depth= 7, min_child_samples=4, n_estimators= 69, num_leaves=11)
estimator.fit(X_train, y_train)
result=estimator.predict(test)
pd.DataFrame({'lgb':result}).to_csv('lgb_result.csv')
