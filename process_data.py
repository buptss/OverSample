#!/usr/bin/env python
# encoding: utf-8

from xgboost import XGBClassifier as xgb
from xgboost import DMatrix
import pandas as pd
from imblearn.metrics import geometric_mean_score
import csv
import heapq
import os
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.externals import joblib
# from settings import reason_dict, target_list, chunk_size, SHOW_FEATURE
import warnings
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# from analysis import analyze_data_proportion, export_pr_auc
chunk_size = 1000000
SHOW_FEATURE = False


def handle_abalone(column_names, filename):
    # 删除此前入库文件
    # os.system("rm -rf ./data/explain.csv")
    directory = "./data/abalone/"
    data = pd.read_csv(directory+filename, names=column_names)
    for label in "MFI":
        data[label] = data["sex"] == label
    del data["sex"]
    data = data[(data["rings"] == 18) | (data["rings"] == 9)]

    y = data.rings.values
    del data["rings"]
    X = data.values.astype(np.float)
    X_resampled, y_resampled = oversample(X, y, method='smote')
    evaluate(X_resampled, y_resampled)
    return


def oversample(x, y, method):
    if method == 'random':
        # 随机过采样
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(x, y)
    elif method == 'smote':
        # SMOTE算法
        X_resampled, y_resampled = SMOTE().fit_resample(x, y)
    elif method == 'adasyn':
        # ADASYN算法
        X_resampled, y_resampled = ADASYN().fit_resample(x, y)
    from collections import Counter
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def evaluate(x, y):
    train_X, test_X, train_y, test_y = train_test_split(x, y)  # splits 75%/25% by default

    # 配置模型
    gbm = xgb(max_depth=3, n_estimators=300, learning_rate=0.05)
    # train
    gbm.fit(train_X, train_y)
    # apply the model to the test and training data
    predicted_test_y = gbm.predict(test_X)
    # predicted_train_y = gbm.predict(train_X)
    # metrics.recall_score()
    print("recall rate:", metrics.recall_score(test_y, predicted_test_y, pos_label=18))
    print("precision rate:", metrics.precision_score(test_y, predicted_test_y, pos_label=18))
    print("f1 score:", metrics.f1_score(test_y, predicted_test_y,pos_label=18))
    print("gmeans:", geometric_mean_score(test_y, predicted_test_y,pos_label=18))
    return


# 功能：主程序
if __name__ == '__main__':
    column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"]
    filename = "abalone.data"
    handle_abalone(column_names, filename)
