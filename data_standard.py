from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy
datasets = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer',
         'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000',
         'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2',
         'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']
for dataset in datasets:
    object = fetch_datasets(data_home='./data/')[dataset]
    X, y = object.data, object.target
    train_X, test_X, train_y, test_y = train_test_split(X, y)  # splits 75%/25% by default
    numpy.savez('./data/zendo_stable/'+dataset+'.npz',train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y)

