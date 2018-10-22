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
from imblearn.datasets import fetch_datasets


# handle each data set
def process(X, y):
    # data split
    train_X, test_X, train_y, test_y = train_test_split(X, y)  # splits 75%/25% by default
    # init sample method
    sample_methods = ['random', 'SMOTE', 'SMOTEBorderline-1', 'SMOTEBorderline-2', 'SVMSMOTE', 'ADASYN', 'No Sample']
    # sample_methods = ['random', 'smote', 'adasyn', 'mwmote']
    dict = {}
    for sample_method in sample_methods:
        # over sample
        X_resampled, y_resampled = oversample(train_X, train_y, method=sample_method)
        # create model
        gbm = xgb(max_depth=3, n_estimators=300, learning_rate=0.05)
        # train model
        gbm.fit(X_resampled, y_resampled)
        # evaluate on test set
        precision, recall, f1, gmean = evaluate(test_X, test_y, sample_method, gbm)
        dict[sample_method] = {"precision": precision, "recall": recall, "f1": f1, "gmean": gmean}
    df = pd.DataFrame(dict)
    # df.set_index(['precision', 'recall', 'gmean', 'f1'], inplace=True)
    df = df.T
    # print(df)
    for index, row in df.iterrows():
        print "&"+index+"&",
        # output precision
        if row["precision"]>=df["precision"].max():
            print r"\textbf{%.3f" % row["precision"] + "}&",
        else:
            print "%.3f" % row["precision"] + "&",

        if row["recall"]>=df["recall"].max():
            print r"\textbf{%.3f" % row["recall"] + "}&",
        else:
            print "%.3f" % row["recall"] + "&",
        if row["f1"]>=df["f1"].max():
            print r"\textbf{%.3f" % row["f1"] + "}&",
        else:
            print "%.3f" % row["f1"] + "&",
        if row["gmean"]>=df["gmean"].max():
            print(r"\textbf{%.3f" % row["gmean"] + r"}\\")
        else:
            print("%.3f" % row["gmean"] + r"\\")
    # evaluate(X, y, "No Sample", gbm)
    return


# involved collected over sample methods
def oversample(x, y, method):
    randomstate = 42
    if method == 'No Sample':
        # 不采样
        return x, y
    elif method == 'random':
        # 随机过采样
        ros = RandomOverSampler(random_state=randomstate)
        X_resampled, y_resampled = ros.fit_resample(x, y)
    elif method == 'SMOTE':
        # SMOTE算法
        X_resampled, y_resampled = SMOTE(random_state=randomstate).fit_resample(x, y)
    elif method == 'SMOTEBorderline-1':
        # BorderlineSmote算法 borderline-1
        X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-1', random_state=randomstate).fit_resample(x, y)
    elif method == 'SMOTEBorderline-2':
        # BorderlineSmote算法 borderline-2
        X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-2', random_state=randomstate).fit_resample(x, y)
    elif method == 'SVMSMOTE':
        # SVMSMOTE算法
        X_resampled, y_resampled = SVMSMOTE(random_state=randomstate).fit_resample(x, y)
    elif method == 'ADASYN':
        # ADASYN算法
        X_resampled, y_resampled = ADASYN(random_state=randomstate).fit_resample(x, y)
    elif method == 'mwmote':
        # MWMOTE算法
        X_resampled, y_resampled = MWMOTE.MWMOTE(x, y, N=1000, return_mode='append')
    # 统计过采样数量
    # from collections import Counter
    # print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


# function: evaluate the oversample effects on the test set
def evaluate(test_X, test_y, model):
    # apply the model to the test and training data
    predicted_test_y = model.predict(test_X)
    # predicted_train_y = gbm.predict(train_X)

    # print("precision rate/recall rate/f1 score/G-means")
    precision = precision_score(test_y, predicted_test_y)
    recall = recall_score(test_y, predicted_test_y)
    f1 = f1_score(test_y, predicted_test_y)
    gmean = geometric_mean_score(test_y, predicted_test_y)
    return precision, recall, f1, gmean


# 功能：主程序
if __name__ == '__main__':
    # column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
    #                 "shell weight", "rings"]
    # filename = "abalone.data"
    # names = ['ecoli']
    names = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer',
             'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000',
             'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2',
             'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']
    for name in names:
        # print('\n\n')
        print(r"\multirow{6}{*}{\textbf{"+name+"}}")
        object = fetch_datasets(data_home='./data/')[name]
        X, y = object.data, object.target
        # print(np.isnan(X).any())
        if np.isnan(X).any():
            print("True")
        # print(np.sum(X == 0))
        # print(X.size)
        # print(name,"%.4f" % (np.sum(X == 0)*1.0/X.size))
        process(X, y)
        print("\hline")
    # handle_abalone(column_names, filename)
    # handle_abalone(column_names, filename)
