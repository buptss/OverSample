#!/usr/bin/env python
# encoding: utf-8

from xgboost import XGBClassifier as xgb
import pandas as pd
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
import MWMOTE
# from imblearn import pipeline as pl

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
    # 取9和18这两列
    data = data[(data["rings"] == 18) | (data["rings"] == 9)]
    pos_label = 18
    y = data.rings.values
    del data["rings"]
    X = data.values.astype(np.float)
    sample_methods = ['random', 'SMOTE', 'SMOTEBorderline-1', 'SMOTEBorderline-2', 'SVMSMOTE', 'ADASYN']
    # sample_methods = ['random', 'smote', 'adasyn', 'mwmote']
    for method in sample_methods:
        print("sample method:", method)
        X_resampled, y_resampled = oversample(X, y, method=method)
        evaluate(X_resampled, y_resampled, pos_label)
    return


def oversample(x, y, method):
    RANDOMSTATE=42
    if method == 'random':
        # 随机过采样
        ros = RandomOverSampler(random_state=RANDOMSTATE)
        X_resampled, y_resampled = ros.fit_resample(x, y)
    elif method == 'SMOTE':
        # SMOTE算法
        X_resampled, y_resampled = SMOTE(random_state=RANDOMSTATE).fit_resample(x, y)
    elif method == 'SMOTEBorderline-1':
        # BorderlineSmote算法 borderline-1
        X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-1', random_state=RANDOMSTATE).fit_resample(x, y)
    elif method == 'SMOTEBorderline-2':
        # BorderlineSmote算法 borderline-2
        X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-2', random_state=RANDOMSTATE).fit_resample(x, y)
    elif method == 'SVMSMOTE':
        # SVMSMOTE算法
        X_resampled, y_resampled = SVMSMOTE(random_state=RANDOMSTATE).fit_resample(x, y)
    elif method == 'ADASYN':
        # ADASYN算法
        X_resampled, y_resampled = ADASYN(random_state=RANDOMSTATE).fit_resample(x, y)
    elif method == 'mwmote':
        # MWMOTE算法
        X_resampled, y_resampled = MWMOTE.MWMOTE(x, y, N=1000, return_mode='append')
    # 统计过采样数量
    # from collections import Counter
    # print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def evaluate(x, y, pos_label):
    train_X, test_X, train_y, test_y = train_test_split(x, y)  # splits 75%/25% by default
    # 配置模型
    gbm = xgb(max_depth=3, n_estimators=300, learning_rate=0.05)
    # train
    gbm.fit(train_X, train_y)
    # apply the model to the test and training data
    predicted_test_y = gbm.predict(test_X)
    # predicted_train_y = gbm.predict(train_X)

    print("precision rate/recall rate/f1 score/G-means")
    print(precision_score(test_y, predicted_test_y, pos_label=pos_label),
          recall_score(test_y, predicted_test_y, pos_label=pos_label),
          f1_score(test_y, predicted_test_y, pos_label=pos_label),
          geometric_mean_score(test_y, predicted_test_y, pos_label=pos_label))
    return


# 功能：主程序
if __name__ == '__main__':
    column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                    "shell weight", "rings"]
    filename = "abalone.data"
    handle_abalone(column_names, filename)
