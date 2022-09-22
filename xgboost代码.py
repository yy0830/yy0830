# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 09:58:13 2022

@author: 86181
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
#load data
dataset = loadtxt("pima-indians-diabets.csv",delimiter=",")
#split data into x and y
X = dataset[:,0:8]
Y = dataset[:,8]
#split data into train and test sets
seed = 7
test_size = 0.33
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
#fit model with training data
model = XGBClassifier()
model.fit(X_train,y_train)
#make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(y_test,predictions)
print("Accuarcy:%.2f%%"%(accuracy*100))
#plot feature importance
plot_importance(model)
pyplot.show()
#   参数调整
model1 = XGBClassifier()
learning_rate = [0.00001,0.001,0.01,0.1,0.2,0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_split=10,shuffle=True,random_state=7)
grid_search=GridSearchCV(model1,param_grid,scoring="neg_log_loss",
                         n_jobs=-1,cv=kfold)
grid_result = grid_search.fit(X,Y)
print("Best:%f using%s"%(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_result_["mean_test_score"]
params = grid_result.cv_results_["params"]
for mean,param in zip(means,params):
    print("%f with %r"%(mean,param))

    