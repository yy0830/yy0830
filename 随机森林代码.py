# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:03:28 2022

@author: 86181
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble as ensemble #ensemble learning
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics


#%%
X,y = make_blobs(n_samples=10000,n_features=10,centers =100, random_state=0)

clf1 = RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores = cross_val_score(clf1,X,y)
print(scores)
print(scores.mean())

#%%
wine = load_wine()
X_train,X_test,Y_train,Y_test = train_test_split(wine.data,wine.target,test_size = 0.3)

clf = RandomForestClassifier(random_state =0)
clf = clf.fit(X_train,Y_train)

score_c = clf.score(X_test,Y_test)
print(score_c)
print(Y_test)

#%%随机森林网格交叉验证调参
param_grid = {
        "criterion":["entropy","gini"],
        "max_depth":[2,3,4,5],
        "n_estimators":[3,5,7],
        "max_features":[0.2,0.3],
        "min_samples_split":[2,3,4]}

rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc,param_grid = param_grid,
                      scoring = "roc_auc",cv=4)
rfc_cv.fit(X_train,Y_train)
#%% 使用随机森林对测试集进行预测
test_est = rfc_cv.predict(X_test)
print("随机森林精确度")
print(metrics.classification_report(test_est,Y_test))
#%%
print(rfc_cv.best_params_)