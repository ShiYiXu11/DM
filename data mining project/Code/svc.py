from sklearn.svm import SVC
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree, preprocessing
svc=SVC()

# train_data = pd.read_csv("../Output/Reduction_Result.csv")
train_data = pd.read_csv("../OriginalData/train.csv")
y=train_data["prognosis"]
X=train_data.drop(columns=["prognosis","id"])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# # tuning
# parameters={
#     'C': [x/100 for x in range(90,110,2)],
#     # 'C': [0.01,0.1,1],
#     'kernel': ['linear','poly','rbf'],
#     'gamma': [x/1000 for x in range(90,110,2)],
#     # 'gamma': [0.1, 1]
#
# }
#
# grid_search_estimator = GridSearchCV(svc, parameters, scoring='accuracy', cv=10)
# print('1')
# grid_search_estimator.fit(X_train, y_train)
# result=grid_search_estimator.cv_results_
# # print the results of all hyper-parameter combinations
# results = pd.DataFrame(grid_search_estimator.cv_results_)
#
# svc_bestScore=grid_search_estimator.best_score_
# svc_bestParams=grid_search_estimator.best_params_
# # print the best parameter setting
# print("best score is {} with params {}".format(svc_bestScore, svc_bestParams))

# ## apply model
drop_features=['diarrhea', 'fatigue', 'nausea', 'lymph_swells', 'muscle_pain', 'rash', 'vomiting', 'pleural_effusion', 'headache', 'sudden_fever', 'joint_pain', 'rigor', 'stiff_neck', 'hypotension', 'chills', 'paralysis', 'gum_bleed', 'gastro_bleeding', 'swelling', 'hyperpyrexia', 'irritability', 'bullseye_rash']
test=pd.read_csv("../OriginalData/test.csv").drop(columns=["id"])
#test=test.drop(columns=drop_features)
estimator=SVC(C=0.9, kernel= 'rbf', gamma=0.102)
estimator.fit(X_train, y_train)
result=estimator.predict(test)
pd.DataFrame({'svc':result}).to_csv('../Output/svc_result.csv')