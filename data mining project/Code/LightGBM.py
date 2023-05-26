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

#train_data = pd.read_csv("../Output/Reduction_Result.csv")
train_data = pd.read_csv("../OriginalData/train.csv")
y=train_data["prognosis"]

X=train_data.drop(columns=["prognosis","id"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# # tuning
# parameters={
#         'num_leaves': range(10,15,2),
#         'n_estimators': range(12,20,2),
#         'min_child_samples' : range(1,5,1),
#         'max_depth' : range(1,10,1),
#         'learning_rate': [x/100 for x in range(9,11,1)],
# }
#
# grid_search_estimator = GridSearchCV(lgb_estimator, parameters, scoring='accuracy', cv=10)
# grid_search_estimator.fit(X_train, y_train)
# result=grid_search_estimator.cv_results_
#
# # print the results of all hyper-parameter combinations
# results = pd.DataFrame(grid_search_estimator.cv_results_)
# xg_bestScore=grid_search_estimator.best_score_
# xg_bestParams=grid_search_estimator.best_params_
# print("best score is {} with params {}".format(xg_bestScore, xg_bestParams))

## apply model
#drop_features=['diarrhea', 'fatigue', 'nausea', 'lymph_swells', 'muscle_pain', 'rash', 'vomiting', 'pleural_effusion', 'headache', 'sudden_fever', 'joint_pain', 'rigor', 'stiff_neck', 'hypotension', 'chills', 'paralysis', 'gum_bleed', 'gastro_bleeding', 'swelling', 'hyperpyrexia', 'irritability', 'bullseye_rash']
test=pd.read_csv("../OriginalData/test.csv").drop(columns=["id"])
# test = test.drop(columns= drop_features)
print(test.shape)

estimator=lgb.LGBMClassifier(learning_rate=0.1, max_depth= 9, min_child_samples=2, n_estimators= 18, num_leaves=10)
estimator.fit(X_train, y_train)
result=estimator.predict(test)
pd.DataFrame({'lgb':result}).to_csv('../Output/lgb_result.csv')
