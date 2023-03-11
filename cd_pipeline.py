import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from pathlib import Path
from CLD import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from EHD import *
from CEDD import *

def get_filepaths(import_path, name):
    # use path module for Windows; adjust for Linux
    # path = Path(sys.argv[2])
    path = Path(import_path+'/'+name)
    # creats an array of file paths in directory
    files = path.glob('*')
    fnames = list(files)
    length = len(fnames)
    fpaths = []
    for f in range(length):
        curr_fname = str(fnames[f].parent) + '/' + str(fnames[f].name)
        fpaths.append(curr_fname)
    return fpaths


def load_data(import_path):
    # store features
    im_features = []
    im_labels = []
    folders = ['adobe', 'alibaba'
                ,
                'amazon', 'apple', 'boa', 'chase', 'dhl'
                'dropbox', 'facebook', 'linkedin', 'microsoft', 'other',
                'paypal', 'wellsfargo', 'yahoo'
               ]
    for name in folders:
        fpaths = get_filepaths(import_path, name)
        for fp in fpaths:
            curr_cld = cld_main(fp)
            curr_ehd = ehdimage(fp)
            curr_cedd = cedd(fp)
            features = np.concatenate((curr_cld, curr_ehd, curr_cedd))
            im_features.append(features)
            if name=='other':
                label = 1
            else:
                label = 0
            im_labels.append(label)
    return im_features, im_labels


def main(X, y, exp_log):
    
    # store datasets in dictionary 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    classifiers = [
        LogisticRegression(max_iter=400)     
        ,SVC()
        ]
    
    names = [
        'LogisticRegression'
        ,'SVC'
        ]
    
    parameters = [
        {'LogisticRegression__C': (0.001, 0.01, 0.1, 1)},
        {'SVC__C': (0.001, 0.01, 0.1, 1)}
        ]
    
    
    for name, classifier, parameter in zip(names, classifiers, parameters):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            (name, classifier)
            ])
        
        grid = GridSearchCV(pipe, param_grid=parameter, n_jobs=-1)
        clf_grid = grid.fit(X_train, y_train)
        score = clf_grid.score(X_test, y_test)
        # identify best estimator
        best_clf = clf_grid.best_estimator_
        
        # fit using best estimator
        train_time_start = datetime.now()
        model = best_clf.fit(X_train, y_train)
        train_time_end = datetime.now()
        
        exp_log.loc[len(exp_log)] = [f"Baseline_features_{name}",
                                     accuracy_score(y_train, model.predict(X_train)), 
                                     #accuracy_score(y_valid, model.predict(X_valid)),
                                     accuracy_score(y_test, model.predict(X_test)),
                                     '--',#roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 
                                     '--'#roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                                     ]


exp_log = pd.DataFrame(columns=["ExpID", "Train Acc", "Test Acc", "Train AUC", "Test AUC"])

import_path = 'phishIRIS_DL_Dataset/phishIRIS_DL_Dataset/train'
name = 'adobe'

feat, labels = load_data(import_path)
model = main(X=feat, y=labels, exp_log=exp_log)
print(exp_log)