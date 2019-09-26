# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm


#: Relative path to the test dataset. This value is used only when the *dtest* cmd argument is None.
# test_dataset = 'datasets/dataset-string-similarity-test.txt'
test_dataset = 'datasets/dataset-string-similarity.txt'

#: float: Similarity threshold on whether sorting on toponym tokens is applied or not. It is triggered on a score
#: below the assigned threshold.
sort_thres = 0.55

thres_range = [30, 71]
thres_weights = [
    [0.25, 0.25, 0.25, 0.25],
    [0.3, 0.3, 0.3, 0.1],
    [0.1, 0.3, 0.3, 0.3],
    [0.3, 0.1, 0.3, 0.3],
    [0.3, 0.1, 0.4, 0.2],
    [0.2, 0.1, 0.2, 0.5],
    [0.1, 0.1, 0.4, 0.4]
]

seed_no = 13


class MLConf:
    """
    This class initializes parameters that correspond to the machine learning part of the framework.

    These variables define the parameter grid for GridSearchCV:

    :cvar SVM_hyperparameters: Defines the search space for SVM.
    :vartype SVM_hyperparameters: :obj:`list`
    :cvar MLP_hyperparameters: Defines the search space for MLP.
    :vartype MLP_hyperparameters: :obj:`dict`
    :cvar DecisionTree_hyperparameters: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters: :obj:`dict`
    :cvar RandomForest_hyperparameters: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters: :obj:`dict`
    :cvar XGBoost_hyperparameters: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters: :obj:`dict`

    These variables define the parameter grid for RandomizedSearchCV where continuous distributions are used for
    continuous parameters (whenever this is feasible):

    :cvar SVM_hyperparameters_dist: Defines the search space for SVM.
    :vartype SVM_hyperparameters_dist: :obj:`dict`
    :cvar MLP_hyperparameters_dist: Defines the search space for MLP.
    :vartype MLP_hyperparameters_dist: :obj:`dict`
    :cvar DecisionTree_hyperparameters_dist: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters_dist: :obj:`dict`
    :cvar RandomForest_hyperparameters_dist: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters_dist: :obj:`dict`
    :cvar XGBoost_hyperparameters_dist: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters_dist: :obj:`dict`
    """

    kfold_parameter = 5  #: int: The number of outer folds that splits the dataset for the k-fold cross-validation.

    #: int: The number of inner folds that splits the dataset for the k-fold cross-validation.
    kfold_inner_parameter = 4

    n_jobs = -1  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.

    train_score = False
    stopping_rounds = 30

    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'grid'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    
    See Also
    --------
    :func:`~src.param_tuning.ParamTuning.getBestClassifier`, :func:`~src.param_tuning.ParamTuning.fineTuneClassifier` 
    Details on available inputs.       
    """
    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when :class:`hyperparams_search_method` equals to 'randomized'.
    max_iter = 300

    clf_static_params = {
        'SVM': {
            'C': 145.30455255834553, 'class_weight': 'balanced', 'degree': 1, 'gamma': 0.3372999022968335,
            'kernel': 'sigmoid', 'max_iter': 3000, 'tol': 0.001,
            'random_state': seed_no
        },
        'DecisionTree': {
            # 'max_features': 8, 'min_samples_split': 0.9, 'min_samples_leaf': 0.1, 'max_depth': 22,
            'max_depth': 56, 'max_features': 5, 'min_samples_leaf': 0.18333333333333335,
            'min_samples_split': 0.8714285714285714,
            'random_state': seed_no,
        },
        'RandomForest': {
            'bootstrap': False, 'criterion': 'gini', 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1,
            'min_samples_split': 2, 'n_estimators': 628,
            'random_state': seed_no, 'n_jobs': n_jobs,  # 'oob_score': True,
        },
        'ExtraTrees': {
            'bootstrap': False, 'criterion': 'gini', 'max_depth': 11, 'max_features': 'sqrt', 'min_samples_leaf': 1,
            'min_samples_split': 3, 'n_estimators': 844,
            'random_state': seed_no, 'n_jobs': n_jobs
        },
        'XGBoost': {
            'colsample_bytree': 0.24017567934980052, 'eta': 0.0324220310237971, 'gamma': 3, 'max_depth': 83,
            'min_child_weight': 8, 'n_estimators': 988, 'subsample': 0.7029480027059571,
            'seed': seed_no, 'nthread': n_jobs
        },
    }

    # These parameters constitute the search space for GridSearchCV in our experiments.
    SVM_hyperparameters = [
        {
            # 'kernel': ['linear', 'rbf'],
            'gamma': [1e-2, 0.1, 1, 10, 100, 'scale'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'tol': [1e-3, 1e-4],
            'dual': [True, False],
            'max_iter': [3000],
            'class_weight': [None, 'balanced', {0: 1, 1: 20}],
        },
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [i for i in range(5, 30, 5)] + [None],
        'min_samples_split': [2, 4, 6, 10, 15, 25],
        'min_samples_leaf': [1, 2, 4, 10],
        # 'min_samples_split': list(np.linspace(0.1, 1, 10)),
        # 'min_samples_leaf': list(np.linspace(0.1, 0.5, 5)),
        'max_features': [list(np.linspace(2, 20, 10))] + ['auto', None],
        'splitter': ('best', 'random'),
    }
    RandomForest_hyperparameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        "n_estimators": [250, 500, 1000],
        'class_weight': [None, 'balanced', {0: 1, 1: 20}],
    }
    XGBoost_hyperparameters = {
        "n_estimators": [500, 1000, 3000],
        'eta': list(np.linspace(0.01, 0.2, 3)),  # 'learning_rate'
        ## avoid overfitting
        # Control the model complexity
        'max_depth': [3, 5, 10, 30, 50, 70, 100],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 5, 10],
        # 'alpha': [1, 10],
        # Add randomness to make training robust to noise
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': list(np.linspace(0.3, 1, 3)),
    }
    MLP_hyperparameters = {
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100),
        'gamma': expon(scale=.1),
        # 'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': ['balanced', None],
        'degree': [1, 2, 3],
        'tol': [1e-3, 1e-4],
        'max_iter': [3000]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 100),
        'min_samples_split': list(np.linspace(0.1, 1, 50)),
        'min_samples_leaf': list(np.linspace(0.1, 0.5, 25)),
        'max_features': sp_randint(1, 11),
    }
    RandomForest_hyperparameters_dist = {
        'bootstrap': [True, False],
        'max_depth': sp_randint(3, 100),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 5),
        'min_samples_split': sp_randint(2, 11),
        "n_estimators": sp_randint(250, 1000),
        'class_weight': ['balanced', None],
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(500, 4000),
        'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'max_depth': sp_randint(3, 100),
        'gamma': sp_randint(1, 5),
        'subsample': truncnorm(0.7, 1),
        'colsample_bytree': truncnorm(0, 1),
        'min_child_weight': sp_randint(1, 10),
    }
    MLP_hyperparameters_dist = {
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }
