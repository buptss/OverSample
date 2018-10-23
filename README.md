# Sparsity Aware Algorithm for Extremely Imbalanced Classification
Sparsity Aware Method for Imbalanced Dataset

# Dataset
UCI Dataset: 
UC Irvine Machine Learning Repository, http://archive.ics.uci.edu/ml/, 2009.

Zendo Dataset:
Ding, Zejin, "Diversified Ensemble Classifiers for Highly Imbalanced Data Learning and their Application in Bioinformatics." Dissertation, Georgia State University, (2011).
   
|   ID | Name     |Repository & Target| Imbalance Ratio| #Samples|#Features| Sparsity Ratio|
| --------   | -----:   | :----: |:----:|:---:|:---:|:---:|
|  1 | ecoli |UCI, target: imU |   8.6:1    |336|7|0.0017|
|2|	optical_digits|	UCI, target: 8|	9.1:1|	5,620|	64|0.4884|
|3|	satimage|	UCI, target: 4|	9.3:1|	6,435|	36|0.0|
|4|	pen_digits|	UCI, target: 5|	9.4:1|	10,992|	16|0.1283|
|5|	abalone	|UCI, target: 7|	9.7:1|	4,177|	10|0.2000|
|6|	sick_euthyroid|	UCI, target: sick euthyroid|	9.8:1|	3,163|	42|0.4360|
|7|	spectrometer|	UCI, target: >=44|	11:1|	531|	93|0.0|
|8|	car_eval_34|	UCI, target: good, v good|	12:1|	1,728|	21|0.7143|
|9|	isolet|	UCI, target: A, B|	12:1|	7,797|	617|0.0036|
|10|	us_crime|	UCI, target: >0.65|	12:1|	1,994|	100|0.0554|
|11|	yeast_ml8|	LIBSVM, target: 8|	13:1|	2,417|	103|0.0|
|12|	scene|	LIBSVM, target: >one label|	13:1|	2,407|	294|0.0115|
|13|	libras_move|	UCI, target: 1|	14:1|	360|	90|0.0001|
|14|	thyroid_sick|	UCI, target: sick|	15:1|	3,772|	52|0.4623|
|15|	coil_2000|	KDD, CoIL, target: minority|	16:1|	9,822|	85|0.5558|
|16|	arrhythmia|	UCI, target: 06|	17:1|	452|	278|0.5352|
|17|	solar_flare_m0|	UCI, target: M->0|	19:1|	1,389|	32|0.6875|
|18|	oil| UCI, target: minority|	22:1|	937|	49|0.0758|
|19|	car_eval_4|	UCI, target: vgood|	26:1|	1,728|	21|0.7143|
|20|	wine_quality|	UCI, wine, target: <=4|	26:1|	4,898|	11|0.0004|
|21|	letter_img|	UCI, target: Z|	26:1|	20,000|	16|0.0262|
|22	|yeast_me2|	UCI, target: ME2|	28:1|	1,484|	8|0.1241|
|23|	webpage|	LIBSVM, w7a, target: minority|	33:1|	34,780|	300|0.9519|
|24|	ozone_level|	UCI, ozone, data|	34:1|	2,536|	72|0.0129|
|25|	mammography|	UCI, target: minority|	42:1|	11,183|	6|0.0000|
|26|	protein_homo|	KDD CUP 2004, minority|	11:1|	145,751|	74|0.0079|
|27|	abalone_19|	UCI, target: 19|	130:1|	4,177|	10|0.2000|




The datasets can be divided into two aspects:

(1) sparsity ratio < 0.5:

|   ID | Name     |Repository & Target| Imbalance Ratio| #Samples|#Features| Sparsity Ratio|
| --------   | -----:   | :----: |:----:|:---:|:---:|:---:|
|  1 | ecoli |UCI, target: imU |   8.6:1    |336|7|0.0017|
|2|	optical_digits|	UCI, target: 8|	9.1:1|	5,620|	64|0.4884|
|3|	satimage|	UCI, target: 4|	9.3:1|	6,435|	36|0.0|
|4|	pen_digits|	UCI, target: 5|	9.4:1|	10,992|	16|0.1283|
|5|	abalone	|UCI, target: 7|	9.7:1|	4,177|	10|0.2000|
|6|	sick_euthyroid|	UCI, target: sick euthyroid|	9.8:1|	3,163|	42|0.4360|
|7|	spectrometer|	UCI, target: >=44|	11:1|	531|	93|0.0|
|9|	isolet|	UCI, target: A, B|	12:1|	7,797|	617|0.0036|
|10|	us_crime|	UCI, target: >0.65|	12:1|	1,994|	100|0.0554|
|11|	yeast_ml8|	LIBSVM, target: 8|	13:1|	2,417|	103|0.0|
|12|	scene|	LIBSVM, target: >one label|	13:1|	2,407|	294|0.0115|
|13|	libras_move|	UCI, target: 1|	14:1|	360|	90|0.0001|
|14|	thyroid_sick|	UCI, target: sick|	15:1|	3,772|	52|0.4623|
|18|	oil| UCI, target: minority|	22:1|	937|	49|0.0758|
|20|	wine_quality|	UCI, wine, target: <=4|	26:1|	4,898|	11|0.0004|
|21|	letter_img|	UCI, target: Z|	26:1|	20,000|	16|0.0262|
|22	|yeast_me2|	UCI, target: ME2|	28:1|	1,484|	8|0.1241|
|24|	ozone_level|	UCI, ozone, data|	34:1|	2,536|	72|0.0129|
|25|	mammography|	UCI, target: minority|	42:1|	11,183|	6|0.0000|
|26|	protein_homo|	KDD CUP 2004, minority|	11:1|	145,751|	74|0.0079|
|27|	abalone_19|	UCI, target: 19|	130:1|	4,177|	10|0.2000|
--------------------- 

(2) sparsity ratio > 0.5:

|   ID | Name     |Repository & Target| Imbalance Ratio| #Samples|#Features| Sparsity Ratio|
| --------   | -----:   | :----: |:----:|:---:|:---:|:---:|
|8|	car_eval_34|	UCI, target: good, v good|	12:1|	1,728|	21|0.7143|
|15|	coil_2000|	KDD, CoIL, target: minority|	16:1|	9,822|	85|0.5558|
|16|	arrhythmia|	UCI, target: 06|	17:1|	452|	278|0.5352|
|17|	solar_flare_m0|	UCI, target: M->0|	19:1|	1,389|	32|0.6875|
|19|	car_eval_4|	UCI, target: vgood|	26:1|	1,728|	21|0.7143|
|23|	webpage|	LIBSVM, w7a, target: minority|	33:1|	34,780|	300|0.9519|
--------------------- 

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

# Classfication Result
https://github.com/buptss/OverSample/blob/master/classification%20result.pdf