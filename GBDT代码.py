# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:21:34 2022

@author: 86181
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#%% 获取训练数据
train_feature = np.genfromtxt("train_feat.txt",dtype=np.float32)
num_feature = len(train_feature[0])
train_feature = pd.DataFrame(train_feature)
train_label = train_feature.iloc[:,num_feature-1]
train_feature = train_feature.iloc[:,0:num_feature-2]
#%% 获取测试数据
test_feature = np.genfromtxt("test_feat.txt",dtype = np.float32)
num_feature = len(test_feature[0])
test_feature = pd.DataFrame(test_feature)
test_label = test_feature.iloc[:,num_feature-1]
test_feature = test_feature.iloc[:,0:num_feature-2]
#%%GBDT模型的建立
gbdt = GradientBoostingRegressor(
        loss="ls",
        learning_rate = 0.1,
        n_estimators = 100,
        subsample = 1,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_depth = 3,
        init = None,
        random_state = None,
        max_features = None,
        alpha = 0.9,
        verbose = 0,
        max_leaf_nodes = None,
        warm_start = False)
gbdt.fit(train_feature,train_label)
pred = gbdt.predict(test_feature)

for i in range(pred.shape[0]):
    print("pred:",pred[i],"label:",test_label[i])
print("均方误差：",np.sqrt(((pred-test_label)**2).mean()))

#%%GBDT调参
gsearch1 = GridSearchCV(estimator = gbdt,param_grid = {"n_estimators":[100,200,300]},scoring="roc_auc"
                                                       ,cv=5)
gsearch1.fit(train_feature,train_label)
#%%
print(gsearch1.best_params_,gsearch1.best_score_)