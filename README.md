<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# OverSample
Informed OverSample Method for Imbalanced Dataset

# Dataset
UCI Dataset: 
UC Irvine Machine Learning Repository, http://archive.ics.uci.edu/ml/, 2009.

# BaseLine
SMOTE:http://www.scs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a.pdf

ADASYN:http://www.ele.uri.edu/faculty/he/PDFfiles/adasyn.pdf

MWMOTE:http://pdfs.semanticscholar.org/c3c0/aaa9f961f0c81d664b0f9a030871de215b79.pdf

BorderlineSMOTE: https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2005-Han-LNCS.pdf

SVMSMOTE：https://www3.nd.edu/~dial/papers/SMCB09.pdf

Random Oversample: 

# Evaluation Principle

|         | Prediction     ||||
| --------   | -----:   | :----: |:----:|:---:|
|         | |Positive      |   Negative    |
| Reality        | Positive      |   True Positive (TP)   | False Positive (FP)|Precision: <img src="http://latex.codecogs.com/gif.latex?\frac{TP}{TP+FP}"/>
|        | Negative      |   False Negative (FN)    | True Negative (TN)|
|        | Recall: <img src="http://latex.codecogs.com/gif.latex?\frac{TP}{TP+FN}"/>      |   Specificity: <img src="http://latex.codecogs.com/gif.latex?\frac{TN}{FP+TN}"/>   | F-value: <img src="http://latex.codecogs.com/gif.latex?\frac{2*recall*precision}{recall+precision}"/>|
|   |   |   |G-mean:<img src="http://latex.codecogs.com/gif.latex?\sqrt{recall * specificity}"/>|
--------------------- 

Precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 

Recall: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.

G-mean: The geometric mean (G-mean) is the root of the product of class-wise sensitivity. 
For binary classification G-mean is the squared root of the product of the sensitivity and specificity.

F1-score: F1 = 2 * (precision * recall) / (precision + recall)
