#!/usr/bin/env python
# encoding: utf-8

IS_LOCAL = True

from xgboost import XGBClassifier as xgb
import pandas as pd
if IS_LOCAL:
    # import sys
    # sys.path.append("..")
    from imbalancedlearn.imblearn.metrics import geometric_mean_score
    # from imbalancedlearn.imblearn.datasets import fetch_datasets
    from imbalancedlearn.imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SparseSMOTE
else:
    from imblearn.metrics import geometric_mean_score
    from imblearn.datasets import fetch_datasets
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE

import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
import MWMOTE
from datetime import datetime
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# from analysis import analyze_data_proportion, export_pr_auc

# control block
# sampling strategy means the minority / majority after the oversample operation.
sampling_strategy = 0.5
chunk_size = 1000000
SHOW_FEATURE = False
SHOW_METRICS = True
SHOW_DURATION_TIME = False
SHOW_AUC_ROC_PLOT = False


def statistics_sample_num(train_X, train_y, X_resampled, y_resampled, sample_method):
    # print the number of sample before and after oversample operation
    before_major = train_X[train_y == -1]
    before_minor = train_X[train_y == 1]
    after_major = X_resampled[y_resampled == -1]
    after_minor = X_resampled[y_resampled == 1]
    after_minority_nonzero_num = len(after_minor[after_minor != 0])
    after_majority_nonzero_num = len(after_major[after_major != 0])
    ratio = after_minority_nonzero_num*1.0/after_majority_nonzero_num
    print(sample_method)
    print("before operation major non-zero num:" + str(len(before_major[before_major != 0])))
    print("before operation minor non-zero num:" + str(len(before_minor[before_minor != 0])))
    print("after operation major non-zero num:" + str(after_majority_nonzero_num))
    print("after operation minor non-zero num:" + str(after_minority_nonzero_num))
    print("after operation non-zero ratio(Minority/Majority):" + str(ratio))
    return


# handle each data set
def process(object):
    train_X, train_y, test_X, test_y = object['train_X'], object['train_y'], object['test_X'], object['test_y']
    # init sample method
    # sample_methods = ['random', 'SMOTE', 'Sparse SMOTE', 'SMOTEBorderline-1', 'SMOTEBorderline-2',
    #                   'SVMSMOTE', 'ADASYN', 'No Sample']
    sample_methods = ['Sparse SMOTE']
    # sample_methods = ['SMOTE']
    # sample_methods = ['random', 'smote', 'adasyn', 'mwmote']
    metrics_dict = {}
    time_info = {}
    for sample_method in sample_methods:
        # before
        before_time = datetime.now()
        # over sample
        X_resampled, y_resampled = oversample(train_X, train_y, method=sample_method)
        statistics_sample_num(train_X, train_y, X_resampled, y_resampled, sample_method)
        # after
        over_time = datetime.now()
        process_time = ((over_time - before_time).microseconds) *1.0/(10**6)
        # print(process_time)
        time_info[sample_method] = "%.3f" % process_time
        # create model
        gbm = xgb(max_depth=3, n_estimators=300, learning_rate=0.01)
        # gbm = xgb(max_depth=3, n_estimators=300, learning_rate=0.01, max_delta_step=0.1)
        # train model
        gbm.fit(X_resampled, y_resampled, eval_metric='auc')
        # evaluate on test set
        precision, recall, f1, gmean, auc_roc, auc_pr, fpr, tpr = evaluate(test_X, test_y, gbm)
        roc_auc = auc(fpr, tpr)
        if SHOW_AUC_ROC_PLOT:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='%s (AUC = %0.2f)' % (sample_method,roc_auc))
        metrics_dict[sample_method] = {"precision": precision, "recall": recall, "f1": f1,
                                       "gmean": gmean, "auc_roc": auc_roc, "auc_pr": auc_pr}
    df = pd.DataFrame(metrics_dict)
    # df.set_index(['precision', 'recall', 'gmean', 'f1'], inplace=True)
    df = df.T
    # print(df)
    if SHOW_METRICS:
        for index, row in df.iterrows():
            print "&"+index+"&",
            # output auc_roc, auc_pr, precision, recall, f1, gmean
            if row["auc_roc"] >= df["auc_roc"].max():
                print r"\textbf{%.3f" % row["auc_roc"] + "}&",
            else:
                print "%.3f" % row["auc_roc"] + "&",
            if row["auc_pr"] >= df["auc_pr"].max():
                print r"\textbf{%.3f" % row["auc_pr"] + "}&",
            else:
                print "%.3f" % row["auc_pr"] + "&",
            if row["precision"] >= df["precision"].max():
                print r"\textbf{%.3f" % row["precision"] + "}&",
            else:
                print "%.3f" % row["precision"] + "&",

            if row["recall"] >= df["recall"].max():
                print r"\textbf{%.3f" % row["recall"] + "}&",
            else:
                print "%.3f" % row["recall"] + "&",
            if row["f1"] >= df["f1"].max():
                print r"\textbf{%.3f" % row["f1"] + "}&",
            else:
                print "%.3f" % row["f1"] + "&",
            if row["gmean"] >= df["gmean"].max():
                print(r"\textbf{%.3f" % row["gmean"] + r"}\\")
            else:
                print("%.3f" % row["gmean"] + r"\\")
    # evaluate(X, y, "No Sample", gbm)
    return time_info


# involved collected over sample methods
def oversample(x, y, method):
    randomstate = 42
    if method == 'No Sample':
        # 不采样
        return x, y
    elif method == 'random':
        # 随机过采样
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=randomstate)
        X_resampled, y_resampled = ros.fit_resample(x, y)
    elif method == 'SMOTE':
        # SMOTE算法
        X_resampled, y_resampled = SMOTE(sampling_strategy=sampling_strategy, random_state=randomstate).fit_resample(x, y)
    elif method == 'Sparse SMOTE':
        # Sparse SMOTE算法
        X_resampled, y_resampled = SparseSMOTE(sampling_strategy=sampling_strategy, random_state=randomstate).fit_resample(x, y)
    elif method == 'SMOTEBorderline-1':
        # BorderlineSmote算法 borderline-1
        X_resampled, y_resampled = BorderlineSMOTE(sampling_strategy=sampling_strategy, kind='borderline-1', random_state=randomstate).fit_resample(x, y)
    elif method == 'SMOTEBorderline-2':
        # BorderlineSmote算法 borderline-2
        X_resampled, y_resampled = BorderlineSMOTE(sampling_strategy=sampling_strategy, kind='borderline-2', random_state=randomstate).fit_resample(x, y)
    elif method == 'SVMSMOTE':
        # SVMSMOTE算法
        X_resampled, y_resampled = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=randomstate).fit_resample(x, y)
    elif method == 'ADASYN':
        # ADASYN算法
        X_resampled, y_resampled = ADASYN(sampling_strategy=sampling_strategy, random_state=randomstate).fit_resample(x, y)
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
    auc_roc = roc_auc_score(test_y, predicted_test_y)
    auc_pr = average_precision_score(test_y, predicted_test_y)
    fpr, tpr, thresholds = roc_curve(test_y, predicted_test_y)

    return precision, recall, f1, gmean, auc_roc, auc_pr, fpr, tpr


def calc_sparsity_ratio(X):
    # calculate sparsity ratio
    print(np.sum(X == 0))
    print(X.size)
    sparsity_ratio = "%.4f" % (np.sum(X == 0)*1.0/X.size)
    return sparsity_ratio


# 功能：主程序
if __name__ == '__main__':
    # column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
    #                 "shell weight", "rings"]
    # filename = "abalone.data"
    # names = ['optical_digits']
    # names = ['ecoli', 'optical_digits']
    # test dataset
    datasets = ['webpage']
    # sparsity ratio >= 0.5
    # datasets = ["car_eval_34", "coil_2000", 'arrhythmia', 'solar_flare_m0','car_eval_4', 'webpage']

    # sparsity ratio < 0.5
    # datasets = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer',
    #             'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick',
    #             'oil', 'wine_quality', 'letter_img', 'yeast_me2',
    #             'ozone_level', 'mammography', 'protein_homo', 'abalone_19']

    # all
    # datasets = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer',
    #          'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000',
    #          'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2',
    #          'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']
    time_dict = {}
    for dataset in datasets:
        if SHOW_AUC_ROC_PLOT:
            plt.cla()
        if SHOW_METRICS:
            print(r"\multirow{6}{*}{\textbf{"+dataset+"}}")
        # object = fetch_datasets(data_home='./data/')[dataset]
        object = np.load('./data/zendo_stable/'+dataset+'.npz')
        time_info = process(object)
        time_dict[dataset] = time_info
        if SHOW_METRICS:
            print("\hline")
        if SHOW_AUC_ROC_PLOT:
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('dataset:' + datasets[0])
            plt.legend(loc="lower right")
            plt.grid()
            plt.savefig(dataset + ".pdf")
    time_df = pd.DataFrame(time_dict)
    time_df = time_df.T
    if SHOW_DURATION_TIME:
        for index, row in time_df.iterrows():
            print index+"&"+row["random"],
            print "&"+row["SMOTE"],
            print "&" + row["SMOTEBorderline-1"],
            print "&" + row["SMOTEBorderline-2"],
            print "&" + row["SVMSMOTE"],
            print "&" + row["ADASYN"] + r"\\"
            print("\hline")

    # handle_abalone(column_names, filename)
    # handle_abalone(column_names, filename)


