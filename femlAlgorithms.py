# -*- coding: utf-8 -*-

from __future__ import print_function
# identifying str and unicode on Python 2, or str on Python 3
from six import string_types

import os, sys
import time
from abc import ABCMeta, abstractmethod
import re
import itertools
import glob
import csv
from text_unidecode import unidecode
from itertools import compress, chain, izip_longest, izip

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline

# We'll use this library to make the display pretty
from tabulate import tabulate

from datasetcreator import strip_accents, LSimilarityVars, lsimilarity_terms, score_per_term, weighted_terms
from helpers import perform_stemming, normalize_str, sorted_nicely, StaticValues
import config


punctuation_regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')


clf_names = [
    # 'SVM':
    [
        LinearSVC, config.MLConf.SVM_hyperparameters, config.MLConf.SVM_hyperparameters_dist
    ],
    # 'DecisionTree':
    [
        DecisionTreeClassifier, config.MLConf.DecisionTree_hyperparameters,
        config.MLConf.DecisionTree_hyperparameters_dist
    ],
    # 'RandomForest':
    [
        RandomForestClassifier, config.MLConf.RandomForest_hyperparameters,
        config.MLConf.RandomForest_hyperparameters_dist
    ],
    # 'MLP':
    [MLPClassifier, config.MLConf.MLP_hyperparameters, config.MLConf.MLP_hyperparameters_dist],
    # 'MLPDummy':
    [MLPClassifier, config.MLConf.MLP_hyperparameters, config.MLConf.MLP_hyperparameters_dist],
    # 'ExtraTrees':
    [
        ExtraTreesClassifier, config.MLConf.RandomForest_hyperparameters,
        config.MLConf.RandomForest_hyperparameters_dist
    ],
    # 'XGBoost':
    [XGBClassifier, config.MLConf.XGBoost_hyperparameters, config.MLConf.XGBoost_hyperparameters_dist]
]


def ascii_transliteration_and_punctuation_strip(s):
    # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
    # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
    # equivalents.

    s = unidecode(strip_accents(s.lower()))
    s = punctuation_regex.sub('', s)
    return s


def transform(strA, strB, sorting=False, stemming=False, canonical=False, delimiter=' ', sort_thres=0.45):
    a = strA.decode('utf8') #.lower()
    b = strB.decode('utf8') #.lower()

    if canonical:
        a = ascii_transliteration_and_punctuation_strip(a)
        b = ascii_transliteration_and_punctuation_strip(b)

    if sorting:
        tmp_a = a.replace(' ', '')
        tmp_b = b.replace(' ', '')

        sim_concatenated = StaticValues.algorithms['damerau_levenshtein'](tmp_a, tmp_b)
        sim_orig = StaticValues.algorithms['damerau_levenshtein'](a, b)
        if sim_concatenated < sort_thres:
            a = " ".join(sorted_nicely(a.split(delimiter)))
            b = " ".join(sorted_nicely(b.split(delimiter)))
        elif sim_concatenated > sim_orig:
            a = tmp_a
            b = tmp_b

    if stemming:
        a = perform_stemming(a)
        b = perform_stemming(b)

    return a, b


def transform_str(s, stemming=False, canonical=False, delimiter=' '):
    a = s.decode('utf8')

    if canonical:
        a = ascii_transliteration_and_punctuation_strip(a)

    if stemming:
        a = perform_stemming(a)

    return a


class PipelineRFE(Pipeline):

    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


class FEMLFeatures:
    no_freq_terms = 200

    def __init__(self):
        pass

    # TODO to_be_removed = "()/.,:!'"  # check the list of chars
    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    @staticmethod
    def contains(str1, str2):
        return all(x in str2 for x in str1.split())

    @staticmethod
    def is_matched(str):
        """
        Finds out how balanced an expression is.
        With a string containing only brackets.

        >>> is_matched('[]()()(((([])))')
        False
        >>> is_matched('[](){{{[]}}}')
        True
        """
        opening = tuple('({[')
        closing = tuple(')}]')
        mapping = dict(zip(opening, closing))
        queue = []

        for letter in str:
            if letter in opening:
                queue.append(mapping[letter])
            elif letter in closing:
                if not queue or letter != queue.pop():
                    return False
        return not queue

    def hasEncoding_err(self, str):
        return self.is_matched(str)

    @staticmethod
    def containsAbbr(str):
        abbr = re.search(r"\b[A-Z]([A-Z\.]{1,}|[sr\.]{1,2})\b", str)
        return '-' if abbr is None else abbr.group()

    @staticmethod
    def containsTermsInParenthesis(str):
        tokens = re.split('[{\[(]', str)
        bflag = True if len(tokens) > 1 else False
        return bflag

    @staticmethod
    def containsDashConnected_words(str):
        """
        Hyphenated words are considered to be:
            * a number of word chars
            * followed by any number of: a single hyphen followed by word chars
        """
        is_dashed = re.search(r"\w+(?:-\w+)+", str)
        return False if is_dashed is None else True

    @staticmethod
    def no_of_words(s1, s2):
        return len(set(s1.split())), len(set(s2.split()))

    @staticmethod
    def containsFreqTerms(s1, s2):
        # specialTerms = dict(a=[], b=[])
        # specialTerms['a'] = filter(lambda x: x in a, freq_terms)
        # specialTerms['b'] = filter(lambda x: x in b, freq_terms)
        # for idx, x in enumerate(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']):
        #     if x in str1: specialTerms['a'].append([idx, x])
        #     if x in str2: specialTerms['b'].append([idx, x])

        s1_flag = any(x in LSimilarityVars.freq_ngrams['tokens'] for x in s1.split())
        s2_flag = any(x in LSimilarityVars.freq_ngrams['tokens'] for x in s2.split())
        return s1_flag, s2_flag

    @staticmethod
    def positionalFreqTerms(s1, s2):
        s1_terms = s1.split()
        s2_terms = s2.split()

        s1_pos = [x in s1_terms for x in list(LSimilarityVars.freq_ngrams['tokens'])[:config.MLConf.pos_freqs]]
        s2_pos = [x in s2_terms for x in list(LSimilarityVars.freq_ngrams['tokens'])[:config.MLConf.pos_freqs]]

        return s1_pos, s2_pos

    @staticmethod
    def ngram_tokens(tokens, ngram=1):
        if tokens < 1: tokens = 1
        return list(itertools.chain.from_iterable([[tokens[i:i + ngram] for i in range(len(tokens) - (ngram - 1))]]))

    def ngrams(str, ngram=1):
       pass

    def _check_size(self, s):
        if not len(s) == 3:
            raise ValueError('expected size 3, got %d' % len(s))

    def containsInPos(self, str1, str2):
        fvec_str1 = []
        fvec_str2 = []

        sep_step = int(round(len(str1) / 3.0))
        fvec_str1.extend(
            [StaticValues.algorithms['damerau_levenshtein'](str1[0:sep_step], str2),
             StaticValues.algorithms['damerau_levenshtein'](str1[sep_step:2*sep_step], str2),
             StaticValues.algorithms['damerau_levenshtein'](str1[2*sep_step:], str2)]
        )

        sep_step = int(round(len(str2) / 3.0))
        fvec_str2.extend(
            [StaticValues.algorithms['damerau_levenshtein'](str1, str2[0:sep_step]),
             StaticValues.algorithms['damerau_levenshtein'](str1, str2[sep_step:2*sep_step]),
             StaticValues.algorithms['damerau_levenshtein'](str1, str2[2*sep_step:])]
        )

        self._check_size(fvec_str1)
        self._check_size(fvec_str2)

        return fvec_str1, fvec_str2

    def get_freqterms(self, encoding):
        input_path = (True, os.path.join(os.getcwd(), 'input/')) \
            if os.path.isdir(os.path.join(os.getcwd(), 'input/')) \
            else (os.path.isdir(os.path.join(os.getcwd(), '../input/')), os.path.join(os.getcwd(), '../input/'))
        if input_path[0]:
            for f in glob.iglob(os.path.join(input_path[1], '*gram*{}{}.csv'.format('_', encoding))):
                gram_type = 'tokens' if 'token' in os.path.basename(os.path.normpath(f)) else 'chars'
                with open(f) as csvfile:
                    print("Loading frequent terms from file {}...".format(f))
                    reader = csv.DictReader(csvfile, fieldnames=["term", "no"], delimiter='\t')
                    _ = reader.fieldnames
                    # go to next line after header
                    next(reader)

                    for i, row in enumerate(reader):
                        if i >= FEMLFeatures.no_freq_terms:
                            break

                        LSimilarityVars.freq_ngrams[gram_type].add(row['term'].decode('utf8'))
            print('Frequent terms loaded.')
        else:
            print("Folder 'input' does not exist")

    def update_weights(self, w):
        if isinstance(w, tuple) and len(w) >= 3:
            del LSimilarityVars.lsimilarity_weights[:]
            LSimilarityVars.lsimilarity_weights.extend(w[:3])


class baseMetrics:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, size, njobs=2, accuracyresults=False):
        self.num_true_predicted_true = [0.0] * size
        self.num_true_predicted_false = [0.0] * size
        self.num_false_predicted_true = [0.0] * size
        self.num_false_predicted_false = [0.0] * size
        self.num_true = 0.0
        self.num_false = 0.0

        self.timer = 0.0
        self.train_timer = 0.0
        self.timers = [0.0] * size

        self.file = None
        self.accuracyresults = accuracyresults

        self.feml_features = FEMLFeatures()

        self.predictedState = {
            'num_true_predicted_true': self.num_true_predicted_true,
            'num_true_predicted_false': self.num_true_predicted_false,
            'num_false_predicted_true': self.num_false_predicted_true,
            'num_false_predicted_false': self.num_false_predicted_false
        }
        self.njobs = njobs

    def __del__(self):
        if self.accuracyresults and self.file is not None and not self.file.closed:
            self.file.close()

    def reset_vars(self):
        self.num_true_predicted_true[:] = [0.0] * len(self.num_true_predicted_true)
        self.num_true_predicted_false[:] = [0.0] * len(self.num_true_predicted_false)
        self.num_false_predicted_true[:] = [0.0] * len(self.num_false_predicted_true)
        self.num_false_predicted_false[:] = [0.0] * len(self.num_false_predicted_false)

        self.timer = 0.0
        self.timers[:] = [0.0] * len(self.timers)

    def preprocessing(self, row):
        if row['res'].upper() == "TRUE": self.num_true += 1.0
        else: self.num_false += 1.0

    def reset(self):
        self.num_true = 0
        self.num_false = 0

    @abstractmethod
    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig', selectable_features=None):
        pass

    def _compute_stats(self, idx, results=False):
        exit_status, acc, pre, rec, f1, timer = 0, -1, -1, -1, -1, 0

        try:
            timer = (self.timers[idx])  # / float(int(self.num_true + self.num_false))) * 50000.0
            acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                  (self.num_true + self.num_false)
            pre = (self.num_true_predicted_true[idx]) / \
                  (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
            rec = (self.num_true_predicted_true[idx]) / \
                  (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
            f1 = 2.0 * ((pre * rec) / (pre + rec))
        except ZeroDivisionError:
            exit_status = 1
            # print("{0} is divided by zero\n".format(StaticValues.methods[idx][0]))

        if results:
            return exit_status, acc, pre, rec, f1, timer

    def _print_stats(self, method, acc, pre, rec, f1, timer):
        # print("Metric = Supervised Classifier :", method)
        # print("Accuracy =", acc)
        # print("Precision =", pre)
        # print("Recall =", rec)
        # print("F1 =", f1)
        # print("Processing time per 50K records =", timer)
        # print("")
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")  # (50K Pairs)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(method, acc, pre, rec, f1, timer))
        # print("")
        sys.stdout.flush()

    def print_stats(self):
        for idx, m in enumerate(StaticValues.methods):
            status, acc, pre, rec, f1, t = self._compute_stats(idx, True)
            if status == 0: self._print_stats(m[0], acc, pre, rec, f1, t)

    def get_stats(self):
        res = {}
        for idx, m in enumerate(StaticValues.methods):
            status, acc, pre, rec, f1, _ = self._compute_stats(idx, True)
            if status == 0: res[m[0]] = [acc, pre, rec, f1]

        return res

    def prediction(self, sim_id, pred_val, real_val, custom_thres):
        result = ""
        var_name = ""

        thres = StaticValues.methods[sim_id - 1][1][custom_thres] if isinstance(custom_thres, string_types) else custom_thres

        if real_val == 1.0:
            if pred_val >= thres:
                var_name = 'num_true_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                result = "\tFALSE"
        else:
            if pred_val >= thres:
                var_name = 'num_false_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                result = "\tFALSE"

        return result, var_name

    def freq_terms_list(self, encoding):
        self.feml_features.get_freqterms(encoding)

    def _perform_feature_selection(self, X_train, y_train, X_test, method, model, no_features_to_keep=12):
        fsupported = None

        if method == 'rfe':
            rfe = RFE(model, n_features_to_select=no_features_to_keep, step=1)
            rfe.fit(X_train, y_train)
            X = rfe.transform(X_train)
            X_t = rfe.transform(X_test)
            fsupported = rfe.support_

            print("{} features selected using Recursive Feature Elimination (RFE).".format(X.shape[1]))
        elif method == 'rfecv':
            rfe = RFECV(model, min_features_to_select=no_features_to_keep, step=2, cv=5)
            rfe.fit(X_train, y_train)
            X = rfe.transform(X_train)
            X_t = rfe.transform(X_test)
            fsupported = rfe.support_

            print("{} feature ranking withRecursive Feature Elimination (RFE) and Cross-Validation (CV).".format(X.shape[1]))
        else:  # default is SelectFromModel ('sfm') module
            # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
            # clf = LassoCV(cv=5, max_iter=2000, n_jobs=2)

            sfm = SelectFromModel(model, threshold=-np.inf, max_features=no_features_to_keep, prefit=False)
            sfm.fit(X_train, y_train)
            X = sfm.transform(X_train)
            X_t = sfm.transform(X_test)
            fsupported = sfm.get_support()

            print("{} features selected using SelectFromModel.".format(X.shape[1]))

        return X, X_t, fsupported

    # Create our function which stores the feature rankings to the ranks dictionary
    def ranking(self, ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))


class calcSotAMetrics(baseMetrics):
    def __init__(self, njobs, accures):
        super(calcSotAMetrics, self).__init__(len(StaticValues.methods), njobs, accures)

    def _generic_evaluator(self, idx, algnm, str1, str2, is_a_match, custom_thres):
        start_time = time.time()
        sim_val = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim_val, is_a_match, custom_thres)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 selectable_features=None):
        tot_res = ""
        flag_true_match = 1.0 if row['res'].upper() == "TRUE" else 0.0

        a, b = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(8, 'jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(2, 'jaro', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(3, 'jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(4, 'jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(11, 'monge_elkan', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(7, 'cosine', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(9, 'strike_a_match', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(12, 'soft_jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(5, 'sorted_winkler', a, b, flag_true_match, custom_thres)
        if permuted: tot_res += self._generic_evaluator(6, 'permuted_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(10, 'skipgram', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(13, 'davies', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(14, 'l_jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(15, 'l_jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)
        if sorting:
            tot_res += self._generic_evaluator(16, 'lsimilarity', a, b, flag_true_match, custom_thres)
            tot_res += self._generic_evaluator(29, 'avg_lsimilarity', a, b, flag_true_match, custom_thres)

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if canonical:
                    file_name += '_canonical'
                if sorting:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))

    def evaluate_sorting(self, row, custom_thres, data_format, stemming=False, permuted=False):
        tot_res = ""
        flag_true_match = 1.0 if row['res'].upper() == "TRUE" else 0.0

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=True, stemming=stemming, canonical=True, thres=custom_thres)

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(8, 'jaccard', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(2, 'jaro', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(3, 'jaro_winkler', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(4, 'jaro_winkler', row['s1'][::-1], row['s2'][::-1], flag_true_match, data_format)
        tot_res += self._generic_evaluator(11, 'monge_elkan', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(7, 'cosine', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(9, 'strike_a_match', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(12, 'soft_jaccard', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(5, 'sorted_winkler', row['s1'], row['s2'], flag_true_match, data_format)
        if permuted: tot_res += self._generic_evaluator(6, 'permuted_winkler', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(10, 'skipgram', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(13, 'davies', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(14, 'l_jaro_winkler', row['s1'], row['s2'], flag_true_match, data_format)
        tot_res += self._generic_evaluator(15, 'l_jaro_winkler', row['s1'][::-1], row['s2'][::-1], flag_true_match, data_format)

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if True:
                    file_name += '_canonical'
                if True:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, row['s1'].encode('utf8'), row['s2'].encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, row['s1'].encode('utf8'), row['s2'].encode('utf8')))


class calcCustomFEML(baseMetrics):
    max_important_features_toshow = 50

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.classifiers = [
            LinearSVC(random_state=0, C=1.0),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(random_state=0, max_depth=100, max_features='auto'),
            RandomForestClassifier(
                n_estimators=250, random_state=0, n_jobs=int(njobs), max_depth=50, oob_score=True, bootstrap=True
            ),
            MLPClassifier(alpha=1, random_state=0),
            # AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=300, random_state=0),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=int(njobs), max_depth=50),
            XGBClassifier(n_estimators=3000, seed=0, nthread=int(njobs)),
        ]
        # self.scores = [[] for _ in range(len(self.classifiers))]
        self.importances = dict()
        self.mlalgs_to_run = StaticValues.classifiers_abbr.keys()

        # To be used within GridSearch
        self.inner_cv = StratifiedKFold(n_splits=config.MLConf.kfold_inner_parameter, shuffle=False,
                                        random_state=config.seed_no)

        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=config.seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

        super(calcCustomFEML, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 selectable_features=None):
        # if row['res'].upper() == "TRUE":
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
        #     else: self.Y2.append(1.0)
        # else:
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
        #     else: self.Y2.append(0.0)
        if row['res'].upper() == "TRUE":
            self.train_Y.append(1.0)
        else:
            self.train_Y.append(0.0)

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)

            self.train_timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     if permuted:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #     else:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            # else:
            if permuted:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
        #     if selectable_features is not None:
        #         self.X1.append(list(compress(chain.from_iterable(tmp_X1), selectable_features)))
        #     else:
        #         self.X1.append(list(chain.from_iterable(tmp_X1)))
        # else:
        if selectable_features is not None:
            self.train_X.append(list(compress(chain.from_iterable(tmp_X2), selectable_features)))
        else:
            self.train_X.append(np.around(list(chain.from_iterable(tmp_X2)), 4).tolist())

        if self.file is None and self.accuracyresults:
            file_name = 'dataset-accuracyresults-sim-metrics'
            if canonical:
                file_name += '_canonical'
            if sorting:
                file_name += '_sorted'
            self.file = open(file_name + '.csv', 'w+')

    def load_test_dataset(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig'):
        if row['res'].upper() == "TRUE":
            self.test_Y.append(1.0)
            self.num_true += 1.0
        else:
            self.test_Y.append(0.0)
            self.num_false += 1.0

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)

            self.timer += (time.time() - start_time)

            if permuted:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append(
                        [sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        self.test_X.append(np.around(list(chain.from_iterable(tmp_X2)), 4).tolist())

    def train_classifiers(self, ml_algs, polynomial=False, standardize=False, fs_method=None, features=None):
        if polynomial:
            self.X1 = PolynomialFeatures().fit_transform(self.X1)
            self.X2 = PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):

        hyperparams_data = list()
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            clf_abbr = StaticValues.classifiers_abbr[name]
            model = self.classifiers[clf_abbr]

            clf = None

            print('Running cv for {}...'.format(name))
            if self.search_method.lower() == 'grid':
                clf = GridSearchCV(
                    clf_names[clf_abbr][0](), clf_names[clf_abbr][1],
                    cv=self.outer_cv, scoring='accuracy', verbose=1,
                    n_jobs=self.n_jobs, return_train_score=config.MLConf.train_score
                )
            # elif self.search_method.lower() == 'hyperband' and clf_key in ['XGBoost', 'Extra-Trees', 'Random Forest']:
            #     HyperbandSearchCV(
            #         clf_val[0](probability=True) if clf_key == 'SVM' else clf_val[0](), clf_val[2].copy().pop('n_estimators'),
            #         resource_param='n_estimators',
            #         min_iter=500 if clf_key == 'XGBoost' else 200,
            #         max_iter=3000 if clf_key == 'XGBoost' else 1000,
            #         cv=self.inner_cv, random_state=seed_no, scoring=score
            #     )
            else:  # randomized is used as default
                clf = RandomizedSearchCV(
                    clf_names[clf_abbr][0](), clf_names[clf_abbr][2],
                    cv=self.outer_cv, scoring='accuracy', verbose=1,
                    n_jobs=self.n_jobs, n_iter=self.n_iter, return_train_score=config.MLConf.train_score
                )
            clf.fit(np.asarray(self.train_X), pd.Series(self.train_Y))

            hyperparams_found = dict()
            hyperparams_found['score'] = clf.best_score_
            hyperparams_found['results'] = clf.cv_results_
            # hyperparams_found['test_len'] = [len(test) for _, test in self.outer_cv.split(X_train, y_train)]
            hyperparams_found['hyperparams'] = clf.best_params_
            hyperparams_found['estimator'] = clf.best_estimator_
            hyperparams_found['classifier'] = name
            hyperparams_found['scorers'] = clf.scorer_

            hyperparams_data.append(hyperparams_found)

            _, best_clf = max(enumerate(hyperparams_data), key=(lambda x: x[1]['score']))
            print('score: {}, hyperparams: {}'.format(best_clf['score'], best_clf['hyperparams']))

            train_time = 0
            predictedL = list()
            tot_features = list()
            print("Training {}...".format(StaticValues.classifiers[clf_abbr]))
            # for X_train, y_train, X_pred, y_pred in izip(
            #         (np.asarray(row, float) for row in (self.X1, self.X2)),
            #         (np.asarray(row, float) for row in (self.Y1, self.Y2)),
            #         (np.asarray(row, float) for row in (self.X2, self.X1)),
            #         ((row for row in (self.Y2, self.Y1)))
            # ):
            start_time = time.time()
            features_supported = [True] * len(StaticValues.featureColumns)
            # if features is not None:
            #     features_supported = [x and y for x, y in zip(features_supported, features)]
            # if fs_method is not None and {'rf', 'et', 'xgboost'}.intersection({name}):
            #     X_train, X_pred, features_supported = self._perform_feature_selection(
            #         X_train, y_train, X_pred, fs_method, model
            #     )
            #     tot_features = [x or y for x, y in izip_longest(features_supported, tot_features, fillvalue=False)]

            # model.fit(X_train, y_train)
            best_clf['estimator'].fit(np.asarray(self.train_X), self.train_Y)
            train_time += (time.time() - start_time)

            start_time = time.time()
            # predictedL += list(model.predict(X_pred))
            predictedL += list(best_clf['estimator'].predict(self.test_X))
            self.timers[clf_abbr] += (time.time() - start_time)

            # if hasattr(model, "feature_importances_"):
            #     if clf_abbr not in self.importances:
            #         self.importances[clf_abbr] = np.zeros(len(StaticValues.featureColumns), dtype=float)
            #
            #     for idx, val in zip([i for i, x in enumerate(features_supported) if x], model.feature_importances_):
            #         self.importances[clf_abbr][idx] += val
            # elif hasattr(model, "coef_"):
            #     if clf_abbr not in self.importances:
            #         self.importances[clf_abbr] = np.zeros(len(StaticValues.featureColumns), dtype=float)
            #
            #     for idx, val in zip([i for i, x in enumerate(features_supported) if x],
            #                         model.coef_.ravel()):
            #         self.importances[clf_abbr][idx] += val
            # # print(model.score(X_pred, y_pred))

            print("Best features discovered: ", end="")
            print(*tot_features, sep=",")
            print("Training took {0:.3f} sec ({1:.3f} min)".format(train_time, train_time / 60.0))
            self.timers[clf_abbr] += self.timer

            print("Matching records...")
            # real = self.Y2 + self.Y1
            real = self.test_Y
            for pos in range(len(real)):
                if real[pos] == 1.0:
                    if predictedL[pos] == 1.0:
                        self.num_true_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tTRUE\n")
                    else:
                        self.num_true_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tFALSE\n")
                else:
                    if predictedL[pos] == 1.0:
                        self.num_false_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tTRUE\n")
                    else:
                        self.num_false_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tFALSE\n")

            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    def print_stats(self):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                continue

            idx = StaticValues.classifiers_abbr[name]
            status, acc, pre, rec, f1, t = self._compute_stats(idx, True)
            if status == 0:
                self._print_stats(StaticValues.classifiers[idx], acc, pre, rec, f1, t)

                if idx not in self.importances or not isinstance(self.importances[idx], np.ndarray):
                    print("The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(
                        name))
                else:
                    importances = self.importances[idx] / 2.0
                    importances = np.ma.masked_equal(importances, 0.0)
                    if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)

                    # indices = np.argsort(importances)[::-1]
                    # for f in range(min(importances.shape[0], self.max_important_features_toshow)):
                    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                    indices = np.argsort(importances.compressed())[::-1][
                              :min(importances.shape[0], self.max_important_features_toshow)]
                    headers = ["name", "score"]
                    print(tabulate(zip(
                        np.asarray(StaticValues.featureColumns, object)[~importances.mask][indices],
                        importances.compressed()[indices]
                    ), headers, tablefmt="simple"))

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print("")
                sys.stdout.flush()


class calcCustomFEMLExtended(baseMetrics):
    max_important_features_toshow = 50
    fterm_feature_size = 10

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        self.best_clf = {}

        self.classifiers = [
            LinearSVC(
                # random_state=0, C=1.0, max_iter=3000
                **config.MLConf.clf_static_params['SVM']
            ),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(
                # random_state=0, max_depth=100, max_features='auto'
                **config.MLConf.clf_static_params['DecisionTree']
            ),
            RandomForestClassifier(
                # default
                # n_estimators=250, max_depth=50, oob_score=True, bootstrap=True
                # optimized
                **config.MLConf.clf_static_params['RandomForest']
            ),
            MLPClassifier(alpha=1, random_state=0),
            # AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=300, random_state=0),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(
                # n_estimators=100, random_state=0, n_jobs=int(njobs), max_depth=50
                **config.MLConf.clf_static_params['ExtraTrees']
            ),
            XGBClassifier(
                # n_estimators=3000, seed=0, nthread=int(njobs)
                **config.MLConf.clf_static_params['XGBoost']
            ),
        ]
        self.scores = [[] for _ in range(len(self.classifiers))]
        self.importances = dict()
        self.mlalgs_to_run = StaticValues.classifiers_abbr.keys()

        # To be used within GridSearch
        self.inner_cv = StratifiedKFold(n_splits=config.MLConf.kfold_inner_parameter, shuffle=False,
                                        random_state=config.seed_no)

        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=config.seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

        super(calcCustomFEMLExtended, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 selectable_features=None):
        # if row['res'].upper() == "TRUE":
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
        #     else: self.Y2.append(1.0)
        # else:
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
        #     else: self.Y2.append(0.0)

        if row['res'].upper() == "TRUE":
            self.train_Y.append(1.0)
        else:
            self.train_Y.append(0.0)

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            if (not flag and not config.MLConf.features_to_build['basic']) or (flag and not config.MLConf.features_to_build['sorted']): continue

            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if flag:
                sim16 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim17 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])
                # sim14 = StaticValues.algorithms['lsimilarity'](a, b)
                sim15 = StaticValues.algorithms['avg_lsimilarity'](a, b)

            self.timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     if permuted:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #     else:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #     if flag: tmp_X1.append([sim16, sim17, sim15])
            # else:
            if permuted:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            if flag: tmp_X2.append([sim16, sim17, sim15])

        # for flag in list({False, True}):
        if config.MLConf.features_to_build['lgm'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            lsim_baseThres = 'avg' if flag else 'simple'

            start_time = time.time()

            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['davies'][lsim_baseThres][0])
            feature17 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'davies', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['skipgram'][lsim_baseThres][0]
            )
            feature18 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'skipgram', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['soft_jaccard'][lsim_baseThres][0]
            )
            feature19 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['strike_a_match'][lsim_baseThres][0]
            )
            feature20 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'strike_a_match', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['cosine'][lsim_baseThres][0]
            )
            feature21 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'cosine', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaccard'][lsim_baseThres][0]
            )
            feature22 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['monge_elkan'][lsim_baseThres][0]
            )
            feature23 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'monge_elkan', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler'][lsim_baseThres][0]
            )
            feature24 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro'][lsim_baseThres][0]
            )
            feature25 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler_r'][lsim_baseThres][0]
            )
            feature26 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'jaro_winkler', flag
            )
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler'][lsim_baseThres][0]
            )
            feature27 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'l_jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler_r'][lsim_baseThres][0]
            )
            feature28 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'l_jaro_winkler', flag
            )

            self.timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     tmp_X1.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
            #                    feature25, feature26, feature27])
            # else:
            tmp_X2.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
                           feature25, feature26, feature27, feature28])

        if config.MLConf.features_to_build['individual'] and sorting:
            start_time = time.time()

            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            method_nm = 'damerau_levenshtein'
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values[method_nm]['avg'][0])
            feature1_1, feature1_2, feature1_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, method_nm)

            # feature8_1, feature8_2, feature8_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'davies')
            # feature9_1, feature9_2, feature9_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'skipgram')
            # feature10_1, feature10_2, feature10_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard')
            # feature11_1, feature11_2, feature11_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'strike_a_match')
            # feature12_1, feature12_2, feature12_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'cosine')
            # feature13_1, feature13_2, feature13_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'monge_elkan')
            # feature14_1, feature14_2, feature14_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler')
            # feature15_1, feature15_2, feature15_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'jaro')
            # feature16_1, feature16_2, feature16_3 = score_per_term(
            #     {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']]},
            #     {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']]},
            #     {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']]},
            #     'jaro_winkler'
            # )

            self.timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     tmp_X1.append([
            #         feature1_1, feature1_2, feature1_3,
            #
            #         # feature8_1, feature8_2, feature8_3,
            #         # feature9_1, feature9_2, feature9_3,
            #         # feature10_1, feature10_2, feature10_3,
            #         # feature11_1, feature11_2, feature11_3,
            #         # feature12_1, feature12_2, feature12_3,
            #         # feature13_1, feature13_2, feature13_3,
            #         # feature14_1, feature14_2, feature14_3,
            #         # feature15_1, feature15_2, feature15_3,
            #         # feature16_1, feature16_2, feature16_3,
            #         # int(feature2_1), int(feature2_2),
            #         # feature3_1, feature3_2,
            #         # int(feature4_1), int(feature4_2),
            #         # int(feature5_1), int(feature5_2)
            #     ])
            #     # tmp_X1.append(map(lambda x: int(x == max(feature6_1)), feature6_1))
            #     # tmp_X1.append(map(lambda x: int(x == max(feature6_2)), feature6_2))
            #     # self.X1[-1].extend(
            #     #     feature7_1[:self.fterm_feature_size/2] + feature7_1[len(feature7_1)/2:self.fterm_feature_size/2] +
            #     #     feature7_2[:self.fterm_feature_size/2] + feature7_2[len(feature7_2)/2:self.fterm_feature_size/2]
            #     # )
            #
            #     if selectable_features is not None:
            #         self.X1.append(list(compress(chain.from_iterable(tmp_X1), selectable_features)))
            #     else:
            #         self.X1.append(list(chain.from_iterable(tmp_X1)))
            # else:
            tmp_X2.append([
                feature1_1, feature1_2, feature1_3,

                # feature8_1, feature8_2, feature8_3,
                # feature9_1, feature9_2, feature9_3,
                # feature10_1, feature10_2, feature10_3,
                # feature11_1, feature11_2, feature11_3,
                # feature12_1, feature12_2, feature12_3,
                # feature13_1, feature13_2, feature13_3,
                # feature14_1, feature14_2, feature14_3,
                # feature15_1, feature15_2, feature15_3,
                # feature16_1, feature16_2, feature16_3,
                # int(feature2_1), int(feature2_2),
                # feature3_1, feature3_2,
                # int(feature4_1), int(feature4_2),
                # int(feature5_1), int(feature5_2)
            ])
                # tmp_X2.append(map(lambda x: int(x == max(feature6_1)), feature6_1))
                # tmp_X2.append(map(lambda x: int(x == max(feature6_2)), feature6_2))

        if config.MLConf.features_to_build['stats'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            # feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
            # feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
            feature1_1, feature1_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
            # feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
            # feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
            feature2_1, feature2_2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
            # feature5_1 = False if len(fterms_s1) == 0 else True
            # feature5_2 = False if len(fterms_s2) == 0 else True
            # feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
            # feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # for x in fterms_s1: feature7_1[x[0]] = 1
            # for x in fterms_s2: feature7_2[x[0]] = 1
            feature3_1, feature3_2 = FEMLFeatures.positionalFreqTerms(row['s1'], row['s2'])

            tmp_X2.append([feature1_1, feature1_2, feature2_1, feature2_2] + feature3_1 + feature3_2)

        if selectable_features is not None:
            self.train_X.append(list(compress(chain.from_iterable(tmp_X2), selectable_features)))
        else:
            self.train_X.append(np.around(list(chain.from_iterable(tmp_X2)), 5).tolist())

        if self.file is None and self.accuracyresults:
            file_name = 'dataset-accuracyresults-sim-metrics'
            if canonical:
                file_name += '_canonical'
            if sorting:
                file_name += '_sorted'
            self.file = open(file_name + '.csv', 'w+')

    def load_test_dataset(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig'):
        if row['res'].upper() == "TRUE":
            self.test_Y.append(1.0)
            self.num_true += 1.0
        else:
            self.test_Y.append(0.0)
            self.num_false += 1.0

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            if (not flag and not config.MLConf.features_to_build['basic']) or (
                    flag and not config.MLConf.features_to_build['sorted']): continue

            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if flag:
                sim16 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim17 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])
                # sim14 = StaticValues.algorithms['lsimilarity'](a, b)
                sim15 = StaticValues.algorithms['avg_lsimilarity'](a, b)

            self.timer += (time.time() - start_time)

            if permuted:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            if flag: tmp_X2.append([sim16, sim17, sim15])

        # for flag in list({False, True}):
        if config.MLConf.features_to_build['lgm'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            lsim_baseThres = 'avg' if flag else 'simple'

            start_time = time.time()

            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['davies'][lsim_baseThres][0])
            feature17 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'davies', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['skipgram'][lsim_baseThres][0]
            )
            feature18 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'skipgram', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['soft_jaccard'][lsim_baseThres][0]
            )
            feature19 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['strike_a_match'][lsim_baseThres][0]
            )
            feature20 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'strike_a_match', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['cosine'][lsim_baseThres][0]
            )
            feature21 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'cosine', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaccard'][lsim_baseThres][0]
            )
            feature22 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['monge_elkan'][lsim_baseThres][0]
            )
            feature23 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'monge_elkan', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler'][lsim_baseThres][0]
            )
            feature24 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro'][lsim_baseThres][0]
            )
            feature25 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler_r'][lsim_baseThres][0]
            )
            feature26 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'jaro_winkler', flag
            )
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler'][lsim_baseThres][0]
            )
            feature27 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'l_jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler_r'][lsim_baseThres][0]
            )
            feature28 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'l_jaro_winkler', flag
            )

            self.timer += (time.time() - start_time)

            tmp_X2.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
                           feature25, feature26, feature27, feature28])

        if config.MLConf.features_to_build['individual'] and sorting:
            start_time = time.time()

            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            method_nm = 'damerau_levenshtein'
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values[method_nm]['avg'][0])
            feature1_1, feature1_2, feature1_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, method_nm)

            # feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
            # feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
            # feature3_1, feature3_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
            # feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
            # feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
            # fterms_s1, fterms_s2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
            # feature5_1 = False if len(fterms_s1) == 0 else True
            # feature5_2 = False if len(fterms_s2) == 0 else True
            # feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
            # feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # for x in fterms_s1: feature7_1[x[0]] = 1
            # for x in fterms_s2: feature7_2[x[0]] = 1

            # feature8_1, feature8_2, feature8_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'davies')
            # feature9_1, feature9_2, feature9_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'skipgram')
            # feature10_1, feature10_2, feature10_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard')
            # feature11_1, feature11_2, feature11_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'strike_a_match')
            # feature12_1, feature12_2, feature12_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'cosine')
            # feature13_1, feature13_2, feature13_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'monge_elkan')
            # feature14_1, feature14_2, feature14_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler')
            # feature15_1, feature15_2, feature15_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, 'jaro')
            # feature16_1, feature16_2, feature16_3 = score_per_term(
            #     {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']]},
            #     {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']]},
            #     {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']]},
            #     'jaro_winkler'
            # )

            self.timer += (time.time() - start_time)

            tmp_X2.append([
                feature1_1, feature1_2, feature1_3,

                # feature8_1, feature8_2, feature8_3,
                # feature9_1, feature9_2, feature9_3,
                # feature10_1, feature10_2, feature10_3,
                # feature11_1, feature11_2, feature11_3,
                # feature12_1, feature12_2, feature12_3,
                # feature13_1, feature13_2, feature13_3,
                # feature14_1, feature14_2, feature14_3,
                # feature15_1, feature15_2, feature15_3,
                # feature16_1, feature16_2, feature16_3,
                # int(feature2_1), int(feature2_2),
                # feature3_1, feature3_2,
                # int(feature4_1), int(feature4_2),
                # int(feature5_1), int(feature5_2)
            ])
            # tmp_X2.append(map(lambda x: int(x == max(feature6_1)), feature6_1))
            # tmp_X2.append(map(lambda x: int(x == max(feature6_2)), feature6_2))

        if config.MLConf.features_to_build['stats'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            # feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
            # feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
            feature1_1, feature1_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
            # feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
            # feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
            feature2_1, feature2_2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
            # feature5_1 = False if len(fterms_s1) == 0 else True
            # feature5_2 = False if len(fterms_s2) == 0 else True
            # feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
            # feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # for x in fterms_s1: feature7_1[x[0]] = 1
            # for x in fterms_s2: feature7_2[x[0]] = 1
            feature3_1, feature3_2 = FEMLFeatures.positionalFreqTerms(row['s1'], row['s2'])

            tmp_X2.append([feature1_1, feature1_2, feature2_1, feature2_2] + feature3_1 + feature3_2)

        self.test_X.append(np.around(list(chain.from_iterable(tmp_X2)), 5).tolist())

    def perform_cv(self, ml_algs, polynomial=False, standardize=False):
        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs

        hyperparams_data = list()
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            clf_abbr = StaticValues.classifiers_abbr[name]
            model = self.classifiers[clf_abbr]
            selector = RFE(model, n_features_to_select=config.MLConf.features_to_select, step=2)
            scaler = MinMaxScaler()
            # scaler = StandardScaler()

            print('Running cv for {}...'.format(name))
            if self.search_method.lower() == 'grid':
                cv = GridSearchCV(
                    clf_names[clf_abbr][0](),
                    clf_names[clf_abbr][1],
                    cv=self.outer_cv, scoring='accuracy', verbose=1, pre_dispatch='2*n_jobs',
                    n_jobs=self.n_jobs, return_train_score=config.MLConf.train_score
                )
            else:  # randomized is used as default
                cv = RandomizedSearchCV(
                    clf_names[clf_abbr][0](),
                    clf_names[clf_abbr][2],
                    cv=self.outer_cv, scoring='accuracy', verbose=1,  pre_dispatch='2*n_jobs',
                    n_jobs=self.n_jobs, n_iter=self.n_iter, return_train_score=config.MLConf.train_score
                )
            # clf.fit(np.asarray(self.train_X), pd.Series(self.train_Y))
            pipe_params = [('scaler', scaler), ('select', selector), ('clf', cv)]
            # pipe_params = [ ('clf', cv)]
            pipe_clf = Pipeline(pipe_params)
            pipe_clf.fit(np.asarray(self.train_X), pd.Series(self.train_Y))

            hyperparams_found = dict()
            # hyperparams_found['score'] = clf.best_score_
            # hyperparams_found['results'] = clf.cv_results_
            # # hyperparams_found['test_len'] = [len(test) for _, test in self.outer_cv.split(X_train, y_train)]
            # hyperparams_found['hyperparams'] = clf.best_params_
            # hyperparams_found['estimator'] = clf.best_estimator_
            # hyperparams_found['classifier'] = name
            # hyperparams_found['scorers'] = clf.scorer_

            hyperparams_found['score'] = pipe_clf.named_steps['clf'].best_score_
            hyperparams_found['results'] = pipe_clf.named_steps['clf'].cv_results_
            hyperparams_found['hyperparams'] = pipe_clf.named_steps['clf'].best_params_
            hyperparams_found['estimator'] = pipe_clf.named_steps['clf'].best_estimator_
            hyperparams_found['classifier'] = name

            hyperparams_data.append(hyperparams_found)

            _, self.best_clf = max(enumerate(hyperparams_data), key=(lambda x: x[1]['score']))
            print('score: {}, hyperparams: {}'.format(self.best_clf['score'], self.best_clf['hyperparams']))

            feature_importances_ = None
            if hasattr(pipe_clf.named_steps['clf'].best_estimator_, 'feature_importances_') or hasattr(pipe_clf.named_steps['clf'].best_estimator_, 'coef_'):
                feature_importances = pipe_clf.named_steps['clf'].best_estimator_.feature_importances_ \
                    if hasattr(pipe_clf.named_steps['clf'].best_estimator_, 'feature_importances_') \
                    else pipe_clf.named_steps['clf'].best_estimator_.coef_

                cols = []
                if config.MLConf.features_to_build['basic']:
                    cols += StaticValues.basicFeatures
                if config.MLConf.features_to_build['sorted']:
                    cols += StaticValues.sortedFeatures
                if config.MLConf.features_to_build['lgm']:
                    cols += StaticValues.lgmFeatures
                if config.MLConf.features_to_build['individual']:
                    cols += StaticValues.individualFeatures
                if config.MLConf.features_to_build['stats']:
                    cols += StaticValues.extraFeatures

                feature_names = np.asarray(cols)  # transformed list to array
                support = pipe_clf.named_steps['select'].support_

                print(feature_importances)
                print('features selected: {}'.format(
                    {k: v for k, v in zip(feature_names[support], feature_importances)}
                ))
                print('features mask: {}'.format(support))
            else: print('Attr "feature_importances_" or "coef_" is not supported!!!')

    def train_classifiers(self, ml_algs, polynomial=False, standardize=False, fs_method=None, features=None):
        # if polynomial:
        #     self.X1 = PolynomialFeatures().fit_transform(self.X1)
        #     self.X2 = PolynomialFeatures().fit_transform(self.X2)
        # if standardize:
        #     # self.X1 = StandardScaler().fit_transform(self.X1)
        #     # self.X2 = StandardScaler().fit_transform(self.X2)
        #     # print(zip(*self.X1)[18][:10], '||', zip(*self.X2)[18][:10])
        #     self.train_X = MinMaxScaler().fit_transform(self.train_X)
        #     self.test_X = MinMaxScaler().fit_transform(self.test_X)
        #     # print(zip(*self.X1)[18][:10], '||', zip(*self.X2)[18][:10])

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs

        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            clf_abbr = StaticValues.classifiers_abbr[name]
            # model = self.classifiers[clf_abbr]
            # selector = RFE(model, n_features_to_select=config.MLConf.features_to_select, step=2)
            # scaler = MinMaxScaler()
            # # scaler = StandardScaler()

            train_time = 0
            predictedL = list()
            tot_features = list()
            print("Training {}...".format(StaticValues.classifiers[clf_abbr]))
            # for X_train, y_train, X_pred, y_pred in izip(
            #         (np.asarray(row, float) for row in (self.X1, self.X2)),
            #         (np.asarray(row, float) for row in (self.Y1, self.Y2)),
            #         (np.asarray(row, float) for row in (self.X2, self.X1)),
            #         (row for row in (self.Y2, self.Y1))
            # ):
            start_time = time.time()
            cols = []
            if config.MLConf.features_to_build['basic']:
                cols += StaticValues.basicFeatures
            if config.MLConf.features_to_build['sorted']:
                cols += StaticValues.sortedFeatures
            if config.MLConf.features_to_build['lgm']:
                cols += StaticValues.lgmFeatures
            if config.MLConf.features_to_build['individual']:
                cols += StaticValues.individualFeatures
            if config.MLConf.features_to_build['stats']:
                cols += StaticValues.extraFeatures
            features_supported = [True] * len(cols)
            # if features is not None:
            #     features_supported = [x and y for x, y in zip(features_supported, features)]
            # if fs_method is not None and set([name]) & {'rf', 'et', 'xgboost'}:
            #     X_train, X_pred, features_supported = self._perform_feature_selection(
            #         X_train, y_train, X_pred, fs_method, model, 11
            #     )
            #     tot_features = [x or y for x, y in izip_longest(features_supported, tot_features, fillvalue=False)]

            # model.fit(X_train, y_train)
            # print('outside cv hyperparams: {}'.format(self.best_clf['estimator'].get_params()))
            self.best_clf['estimator'].fit(self.train_X, self.train_Y)
            # print(best_clf['estimator'].feature_importances_)
            train_time += (time.time() - start_time)

            start_time = time.time()
            # predictedL += list(model.predict(X_pred))
            # self.test_X = pipe_clf.transform(self.test_X)
            predictedL += list(self.best_clf['estimator'].predict(self.test_X))
            self.timers[clf_abbr] += (time.time() - start_time)

            if hasattr(self.best_clf['estimator'], "feature_importances_"):
                # self.importances[i] += model.feature_importances_
                if clf_abbr not in self.importances:
                    self.importances[clf_abbr] = np.zeros(len(cols), dtype=float)

                for idx, val in zip([i for i, x in enumerate(features_supported) if x], self.best_clf['estimator'].feature_importances_):
                    self.importances[clf_abbr][idx] += val
            elif hasattr(self.best_clf['estimator'], "coef_"):
                if clf_abbr in self.importances:
                    self.importances[clf_abbr] += self.best_clf['estimator'].coef_.ravel()
                else:
                    self.importances[clf_abbr] = self.best_clf['estimator'].coef_.ravel()
            # self.scores[i].append(model.score(np.array(pred_X), np.array(pred_Y)))
            # if name in ['rf']:
            #     print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(
            #         model.score(X_train, y_train),
            #         model.oob_score_,
            #         model.score(X_pred, y_pred))
            #     )

            print("Best Features discovered: ", end="")
            print(*tot_features, sep=",")
            print("Training took {0:.3f} sec ({1:.3f} min)".format(train_time, train_time / 60.0))
            self.timers[clf_abbr] += self.timer

            print("Matching records...")
            # real = self.Y2 + self.Y1
            real = self.test_Y
            for pos in range(len(real)):
                if real[pos] == 1.0:
                    if predictedL[pos] == 1.0:
                        self.num_true_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tTRUE\n")
                    else:
                        self.num_true_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tFALSE\n")
                else:
                    if predictedL[pos] == 1.0:
                        self.num_false_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tTRUE\n")
                    else:
                        self.num_false_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tFALSE\n")

            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    def print_stats(self):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                continue

            idx = StaticValues.classifiers_abbr[name]
            status, acc, pre, rec, f1, t = self._compute_stats(idx, True)
            if status == 0:
                self._print_stats(StaticValues.classifiers[idx], acc, pre, rec, f1, t)

                if idx not in self.importances or not isinstance(self.importances[idx], np.ndarray):
                    print("The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(
                        name))
                else:
                    cols = []
                    if config.MLConf.features_to_build['basic']:
                        cols += StaticValues.basicFeatures
                    if config.MLConf.features_to_build['sorted']:
                        cols += StaticValues.sortedFeatures
                    if config.MLConf.features_to_build['lgm']:
                        cols += StaticValues.lgmFeatures
                    if config.MLConf.features_to_build['individual']:
                        cols += StaticValues.individualFeatures
                    if config.MLConf.features_to_build['stats']:
                        cols += StaticValues.extraFeatures

                    importances = self.importances[idx]
                    importances = np.ma.masked_equal(importances, 0.0)
                    if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)

                    indices = np.argsort(importances.compressed())[::-1][
                              :min(importances.compressed().shape[0], self.max_important_features_toshow)]
                    headers = ["name", "score"]
                    print(tabulate(zip(
                        np.asarray(cols, object)[~importances.mask][indices],
                        importances.compressed()[indices]
                    ), headers, tablefmt="simple"))

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print("")
                sys.stdout.flush()

    def debug_stats(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        print('')

        cols = []
        if config.MLConf.features_to_build['basic']:
            cols += StaticValues.basicFeatures
        if config.MLConf.features_to_build['sorted']:
            cols += StaticValues.sortedFeatures
        if config.MLConf.features_to_build['lgm']:
            cols += StaticValues.lgmFeatures
        if config.MLConf.features_to_build['individual']:
            cols += StaticValues.individualFeatures
        if config.MLConf.features_to_build['stats']:
            cols += StaticValues.extraFeatures

        df = pd.DataFrame(np.array(self.X1).reshape(-1, len(cols)), columns=cols)
        # with pd.option_context('display.max_columns', None):
        output_f = './output/X1_train_stats.csv'
        df.describe().T.to_csv(output_f)
        with open(output_f, 'a') as f:
            print("Existence of null values in X1_train: {}".format(df.isnull().values.any()))
            f.write("\nExistence of null values in X1_train: {}\n".format(df.isnull().values.any()))
            print(df.mode(axis=0, dropna=False).T)
            # f.write("Highest freq values per column in X1_train\n")
            # df.mode(axis=0, dropna=False).T.to_csv(f, header=False)

        df = pd.DataFrame(np.array(self.X2).reshape(-1, len(cols)), columns=cols)
        output_f = './output/X2_train_stats.csv'
        df.describe().T.to_csv(output_f)
        with open(output_f, 'a') as f:
            print("Existence of null values in X2_train: {}".format(df.isnull().values.any()))
            f.write("\nExistence of null values in X2_train: {}\n".format(df.isnull().values.any()))
            print(df.mode(axis=0, dropna=False).transpose())
            # f.write("\nHighest freq values per column in X2_train\n")
            # df.mode(axis=0, dropna=False).transpose().to_csv(f, header=False)


class calcWithCustomHyperparams(baseMetrics):
    max_important_features_toshow = 50
    fname = ''

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.classifiers = [
            LinearSVC(
                # random_state=0, C=1.0, max_iter=3000
                **config.MLConf.clf_static_params['SVM']
            ),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(
                # random_state=0, max_depth=100, max_features='auto'
                **config.MLConf.clf_static_params['DecisionTree']
            ),
            RandomForestClassifier(
                # default
                # n_estimators=250, max_depth=50, oob_score=True, bootstrap=True
                # optimized
                **config.MLConf.clf_static_params['RandomForest']
            ),
            MLPClassifier(alpha=1, random_state=0),
            # AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=300, random_state=0),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(
                # n_estimators=100, random_state=0, n_jobs=int(njobs), max_depth=50
                **config.MLConf.clf_static_params['ExtraTrees']
            ),
            XGBClassifier(
                # n_estimators=3000, seed=0, nthread=int(njobs)
                **config.MLConf.clf_static_params['XGBoost']
            ),
        ]
        # self.scores = [[] for _ in range(len(self.classifiers))]
        self.importances = dict()
        self.mlalgs_to_run = StaticValues.classifiers_abbr.keys()

        # To be used within GridSearch
        self.inner_cv = StratifiedKFold(n_splits=config.MLConf.kfold_inner_parameter, shuffle=False,
                                        random_state=config.seed_no)

        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=config.seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

        super(calcWithCustomHyperparams, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 selectable_features=None):
        # if row['res'].upper() == "TRUE":
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
        #     else: self.Y2.append(1.0)
        # else:
        #     if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
        #     else: self.Y2.append(0.0)
        if row['res'].upper() == "TRUE":
            self.train_Y.append(1.0)
        else:
            self.train_Y.append(0.0)

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            if (not flag and not config.MLConf.features_to_build['basic']) or (
                    flag and not config.MLConf.features_to_build['sorted']): continue

            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if flag:
                sim16 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim17 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])
                # sim14 = StaticValues.algorithms['lsimilarity'](a, b)
                sim15 = StaticValues.algorithms['avg_lsimilarity'](a, b)

            self.train_timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     if permuted:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #     else:
            #         if flag: tmp_X1.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            #         else: tmp_X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            # else:
            if permuted:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag: tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else: tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            if flag: tmp_X2.append([sim16, sim17, sim15])

        # for flag in list({False, True}):
        if config.MLConf.features_to_build['lgm'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            lsim_baseThres = 'avg' if flag else 'simple'

            start_time = time.time()

            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['davies'][lsim_baseThres][0])
            feature17 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'davies', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['skipgram'][lsim_baseThres][0]
            )
            feature18 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'skipgram', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['soft_jaccard'][lsim_baseThres][0]
            )
            feature19 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['strike_a_match'][lsim_baseThres][0]
            )
            feature20 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'strike_a_match', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['cosine'][lsim_baseThres][0]
            )
            feature21 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'cosine', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaccard'][lsim_baseThres][0]
            )
            feature22 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['monge_elkan'][lsim_baseThres][0]
            )
            feature23 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'monge_elkan', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler'][lsim_baseThres][0]
            )
            feature24 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro'][lsim_baseThres][0]
            )
            feature25 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler_r'][lsim_baseThres][0]
            )
            feature26 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'jaro_winkler', flag
            )
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler'][lsim_baseThres][0]
            )
            feature27 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'l_jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler_r'][lsim_baseThres][0]
            )
            feature28 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'l_jaro_winkler', flag
            )

            self.timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     tmp_X1.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
            #                    feature25, feature26, feature27])
            # else:
            tmp_X2.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
                           feature25, feature26, feature27, feature28])

        if config.MLConf.features_to_build['individual'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)
            start_time = time.time()

            method_nm = 'damerau_levenshtein'
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values[method_nm]['avg'][0])
            feature1_1, feature1_2, feature1_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, method_nm)

            self.timer += (time.time() - start_time)

            tmp_X2.append([
                feature1_1, feature1_2, feature1_3,

                # feature8_1, feature8_2, feature8_3,
                # feature9_1, feature9_2, feature9_3,
                # feature10_1, feature10_2, feature10_3,
                # feature11_1, feature11_2, feature11_3,
                # feature12_1, feature12_2, feature12_3,
                # feature13_1, feature13_2, feature13_3,
                # feature14_1, feature14_2, feature14_3,
                # feature15_1, feature15_2, feature15_3,
                # feature16_1, feature16_2, feature16_3,
                # int(feature2_1), int(feature2_2),
                # feature3_1, feature3_2,
                # int(feature4_1), int(feature4_2),
                # int(feature5_1), int(feature5_2)
            ])

        if config.MLConf.features_to_build['stats'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)
            # feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
            # feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
            feature1_1, feature1_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
            # feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
            # feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
            feature2_1, feature2_2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
            # feature5_1 = False if len(fterms_s1) == 0 else True
            # feature5_2 = False if len(fterms_s2) == 0 else True
            # feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
            # feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # for x in fterms_s1: feature7_1[x[0]] = 1
            # for x in fterms_s2: feature7_2[x[0]] = 1
            feature3_1, feature3_2 = FEMLFeatures.positionalFreqTerms(row['s1'], row['s2'])

            tmp_X2.append([feature1_1, feature1_2, feature2_1, feature2_2] + feature3_1 + feature3_2)

        # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
        #     if selectable_features is not None:
        #         self.X1.append(list(compress(chain.from_iterable(tmp_X1), selectable_features)))
        #     else:
        #         self.X1.append(list(chain.from_iterable(tmp_X1)))
        # else:
        if selectable_features is not None:
            self.train_X.append(list(compress(chain.from_iterable(tmp_X2), selectable_features)))
        else:
            self.train_X.append(np.around(list(chain.from_iterable(tmp_X2)), 5).tolist())

        if not self.fname:
            self.fname = 'results-evaluation'
            if canonical:
                self.fname += '_canonical'
            if sorting:
                self.fname += '_sorted'

    def load_test_dataset(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig'):
        if row['res'].upper() == "TRUE":
            self.test_Y.append(1.0)
            self.num_true += 1.0
        else:
            self.test_Y.append(0.0)
            self.num_false += 1.0

        tmp_X1, tmp_X2 = [], []
        for flag in list({False, sorting}):
            if (not flag and not config.MLConf.features_to_build['basic']) or (
                    flag and not config.MLConf.features_to_build['sorted']): continue

            a, b = transform(row['s1'], row['s2'], sorting=flag, stemming=stemming, canonical=flag)

            start_time = time.time()

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not flag: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if flag:
                sim16 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim17 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])
                # sim14 = StaticValues.algorithms['lsimilarity'](a, b)
                sim15 = StaticValues.algorithms['avg_lsimilarity'](a, b)

            self.timer += (time.time() - start_time)

            if permuted:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append(
                        [sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                if flag:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
                else:
                    tmp_X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            if flag: tmp_X2.append([sim16, sim17, sim15])

        if config.MLConf.features_to_build['lgm'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)

            lsim_baseThres = 'avg' if flag else 'simple'

            start_time = time.time()

            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['davies'][lsim_baseThres][0])
            feature17 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'davies', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['skipgram'][lsim_baseThres][0]
            )
            feature18 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'skipgram', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['soft_jaccard'][lsim_baseThres][0]
            )
            feature19 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'soft_jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['strike_a_match'][lsim_baseThres][0]
            )
            feature20 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'strike_a_match', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['cosine'][lsim_baseThres][0]
            )
            feature21 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'cosine', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaccard'][lsim_baseThres][0]
            )
            feature22 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaccard', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['monge_elkan'][lsim_baseThres][0]
            )
            feature23 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'monge_elkan', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler'][lsim_baseThres][0]
            )
            feature24 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro'][lsim_baseThres][0]
            )
            feature25 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'jaro', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['jaro_winkler_r'][lsim_baseThres][0]
            )
            feature26 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'jaro_winkler', flag
            )
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler'][lsim_baseThres][0]
            )
            feature27 = weighted_terms(baseTerms, mismatchTerms, specialTerms, 'l_jaro_winkler', flag)
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values['l_jaro_winkler_r'][lsim_baseThres][0]
            )
            feature28 = weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                'l_jaro_winkler', flag
            )

            self.timer += (time.time() - start_time)

            # if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            #     tmp_X1.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
            #                    feature25, feature26, feature27])
            # else:
            tmp_X2.append([feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24,
                           feature25, feature26, feature27, feature28])

        if config.MLConf.features_to_build['individual'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)
            start_time = time.time()

            method_nm = 'damerau_levenshtein'
            baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
                row['s1'], row['s2'], LSimilarityVars.per_metric_optimal_values[method_nm]['avg'][0])
            feature1_1, feature1_2, feature1_3 = score_per_term(baseTerms, mismatchTerms, specialTerms, method_nm)

            self.timer += (time.time() - start_time)

            tmp_X2.append([
                feature1_1, feature1_2, feature1_3,

                # feature8_1, feature8_2, feature8_3,
                # feature9_1, feature9_2, feature9_3,
                # feature10_1, feature10_2, feature10_3,
                # feature11_1, feature11_2, feature11_3,
                # feature12_1, feature12_2, feature12_3,
                # feature13_1, feature13_2, feature13_3,
                # feature14_1, feature14_2, feature14_3,
                # feature15_1, feature15_2, feature15_3,
                # feature16_1, feature16_2, feature16_3,
                # int(feature2_1), int(feature2_2),
                # feature3_1, feature3_2,
                # int(feature4_1), int(feature4_2),
                # int(feature5_1), int(feature5_2)
            ])

        if config.MLConf.features_to_build['stats'] and sorting:
            row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming,
                                             canonical=canonical)
            # feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
            # feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
            feature1_1, feature1_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
            # feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
            # feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
            feature2_1, feature2_2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
            # feature5_1 = False if len(fterms_s1) == 0 else True
            # feature5_2 = False if len(fterms_s2) == 0 else True
            # feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
            # feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens'] | LSimilarityVars.freq_ngrams['chars']))
            # for x in fterms_s1: feature7_1[x[0]] = 1
            # for x in fterms_s2: feature7_2[x[0]] = 1
            feature3_1, feature3_2 = FEMLFeatures.positionalFreqTerms(row['s1'], row['s2'])

            tmp_X2.append([feature1_1, feature1_2, feature2_1, feature2_2] + feature3_1 + feature3_2)

        new_features = np.around(list(chain.from_iterable(tmp_X2)), 5)
        self.test_X.append(new_features.tolist())

    def train_classifiers(self, ml_algs, polynomial=False, standardize=False, fs_method=None, features=None):
        if polynomial:
            self.X1 = PolynomialFeatures().fit_transform(self.X1)
            self.X2 = PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):

        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            clf_abbr = StaticValues.classifiers_abbr[name]
            model = self.classifiers[clf_abbr]

            train_time = 0
            predictedL = list()
            tot_features = list()
            print("Training {}...".format(StaticValues.classifiers[clf_abbr]))
            # for X_train, y_train, X_pred, y_pred in izip(
            #         (np.asarray(row, float) for row in (self.X1, self.X2)),
            #         (np.asarray(row, float) for row in (self.Y1, self.Y2)),
            #         (np.asarray(row, float) for row in (self.X2, self.X1)),
            #         ((row for row in (self.Y2, self.Y1)))
            # ):
            start_time = time.time()
            cols = []
            if config.MLConf.features_to_build['basic']:
                cols += StaticValues.basicFeatures
            if config.MLConf.features_to_build['sorted']:
                cols += StaticValues.sortedFeatures
            if config.MLConf.features_to_build['lgm']:
                cols += StaticValues.lgmFeatures
            if config.MLConf.features_to_build['individual']:
                cols += StaticValues.individualFeatures
            if config.MLConf.features_to_build['stats']:
                cols += StaticValues.extraFeatures
            feature_names = np.asarray(cols)

            # features_supported = [True] * len(cols)
            # if features is not None:
            #     features_supported = [x and y for x, y in zip(features_supported, features)]
            # if fs_method is not None and {'rf', 'et', 'xgboost'}.intersection({name}):
            #     X_train, X_pred, features_supported = self._perform_feature_selection(
            #         X_train, y_train, X_pred, fs_method, model
            #     )
            #     tot_features = [x or y for x, y in izip_longest(features_supported, tot_features, fillvalue=False)]

            # selector = RFE(model, n_features_to_select=config.MLConf.features_to_select, step=2)
            scaler = MinMaxScaler()
            # scaler = StandardScaler()

            pipe_params = None
            # TODO check why hasattr cannot find feature_importances_
            # if hasattr(model, 'feature_importances_') or \
            #         isinstance(getattr(type(model), 'feature_importances_', None), property) or \
            #         hasattr(model, 'coef_') or \
            #         isinstance(getattr(type(model), 'coef_', None), property):
            #     print('feature_importances found for clf {}'.format(name))
            selector = RFE(model, n_features_to_select=config.MLConf.features_to_select, step=2)
            pipe_params = [('scaler', scaler), ('clf', selector)]
            # else:
            #     pipe_params = [('scaler', scaler), ('clf', model)]
            pipe_clf = Pipeline(pipe_params)
            pipe_clf.fit(self.train_X, self.train_Y)
            # print(pipe_clf.named_steps['clf'].support_)
            # model.fit(np.asarray(self.train_X), self.train_Y)
            train_time += (time.time() - start_time)

            start_time = time.time()

            predictedL += list(pipe_clf.named_steps['clf'].predict(self.test_X))
            # predictedL += list(best_clf['estimator'].predict(self.test_X))
            self.timers[clf_abbr] += (time.time() - start_time)

            # print(pipe_clf.named_steps['clf'].ranking_)
            if hasattr(pipe_clf.named_steps['clf'].estimator_, "feature_importances_"):
                if clf_abbr not in self.importances:
                    self.importances[clf_abbr] = np.zeros(len(cols), dtype=float)

                feature_importances = pipe_clf.named_steps['clf'].estimator_.feature_importances_
                support = pipe_clf.named_steps['clf'].support_
                for k, v in zip(feature_names[support], feature_importances):
                    self.importances[clf_abbr][cols.index(k)] += v
            elif hasattr(pipe_clf.named_steps['clf'].estimator_, "coef_"):
                if clf_abbr not in self.importances:
                    self.importances[clf_abbr] = np.zeros(len(cols), dtype=float)

                feature_importances = pipe_clf.named_steps['clf'].estimator_.coef_.ravel()
                support = pipe_clf.named_steps['clf'].support_
                for k, v in zip(feature_names[support], feature_importances):
                    self.importances[clf_abbr][cols.index(k)] += v
            # # print(model.score(X_pred, y_pred))

            print("Best features discovered: ", end="")
            print(*tot_features, sep=",")
            print("Training took {0:.3f} sec ({1:.3f} min)".format(train_time, train_time / 60.0))
            self.timers[clf_abbr] += self.timer

            if self.accuracyresults: self.file = open('{}_{}.csv'.format(self.fname, name), 'w+')

            print("Matching records...")
            # real = self.Y2 + self.Y1
            real = self.test_Y
            for pos in range(len(real)):
                if real[pos] == 1.0:
                    if predictedL[pos] == 1.0:
                        self.num_true_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tTRUE\n")
                    else:
                        self.num_true_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tFALSE\n")
                else:
                    if predictedL[pos] == 1.0:
                        self.num_false_predicted_true[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tTRUE\n")
                    else:
                        self.num_false_predicted_false[clf_abbr] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tFALSE\n")

            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            if self.accuracyresults and not self.file.closed: self.file.close()

    def print_stats(self):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                continue

            idx = StaticValues.classifiers_abbr[name]
            status, acc, pre, rec, f1, t = self._compute_stats(idx, True)
            if status == 0:
                self._print_stats(StaticValues.classifiers[idx], acc, pre, rec, f1, t)

                if idx not in self.importances or not isinstance(self.importances[idx], np.ndarray):
                    print("The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(
                        name))
                else:
                    cols = []
                    if config.MLConf.features_to_build['basic']:
                        cols += StaticValues.basicFeatures
                    if config.MLConf.features_to_build['sorted']:
                        cols += StaticValues.sortedFeatures
                    if config.MLConf.features_to_build['lgm']:
                        cols += StaticValues.lgmFeatures
                    if config.MLConf.features_to_build['individual']:
                        cols += StaticValues.individualFeatures
                    if config.MLConf.features_to_build['stats']:
                        cols += StaticValues.extraFeatures
                    importances = self.importances[idx]
                    importances = np.ma.masked_equal(importances, 0.0)
                    if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)

                    # indices = np.argsort(importances)[::-1]
                    # for f in range(min(importances.shape[0], self.max_important_features_toshow)):
                    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                    indices = np.argsort(importances.compressed())[::-1][
                              :min(importances.shape[0], self.max_important_features_toshow)]
                    headers = ["name", "score"]
                    print(tabulate(zip(
                        np.asarray(cols, object)[~importances.mask][indices],
                        importances.compressed()[indices]
                    ), headers, tablefmt="simple"))

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print("")
                sys.stdout.flush()


class calcDLearning(baseMetrics):
    pass


class calcSotAML(baseMetrics):
    pass


class calcLSimilarities(baseMetrics):
    def __init__(self, njobs, accures):
        super(calcLSimilarities, self).__init__(len(StaticValues.methods), njobs, accures)

    def _generic_evaluator(self, idx, lgm_metric, str1, str2, is_a_match, custom_thres):
        tot_res = ""

        for alg_info in [[13, 'avg_lsimilarity']]:
            start_time = time.time()
            sim_val = StaticValues.algorithms[alg_info[1]](str1, str2, method=lgm_metric)
            res, varnm = self.prediction(idx + alg_info[0], sim_val, is_a_match, custom_thres)
            self.timers[idx + alg_info[0] - 1] += (time.time() - start_time)
            self.predictedState[varnm][idx + alg_info[0] - 1] += 1.0
            tot_res += res

        return tot_res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 features=None, selectable_features=None):
        tot_res = ""
        flag_true_match = 1.0 if row['res'].upper() == "TRUE" else 0.0

        a, b = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        tot_res += self._generic_evaluator(16, 'damerau_levenshtein', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(21, 'jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(17, 'jaro', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(18, 'jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(19, 'jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(24, 'monge_elkan', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(20, 'cosine', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(22, 'strike_a_match', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(25, 'soft_jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(23, 'skipgram', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(26, 'davies', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(27, 'l_jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(28, 'l_jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if canonical:
                    file_name += '_canonical'
                if sorting:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))


class testMetrics(baseMetrics):
    def __init__(self, njobs, accures):
        super(testMetrics, self).__init__(len(StaticValues.methods), njobs, accures)

    def _generic_evaluator(self, idx, sim_metric, baseTerms, mismatchTerms, specialTerms, is_a_match, custom_thres):
        start_time = time.time()

        sim_val = weighted_terms(baseTerms, mismatchTerms, specialTerms, sim_metric, averaged=True, test_mode=True)
        res, varnm = self.prediction(idx, sim_val, is_a_match, custom_thres)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, custom_thres='orig',
                 term_split_thres=0.55):
        tot_res = ""
        flag_true_match = 1.0 if row['res'].upper() == "TRUE" else 0.0

        a, b = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(a, b, term_split_thres)
        rbaseTerms = {
            'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
            'len': baseTerms['len'], 'char_len': baseTerms['char_len']
        }
        rmismatchTerms = {
            'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
            'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']
        }
        rspecialTerms = {
            'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
            'len': specialTerms['len'], 'char_len': specialTerms['char_len']
        }

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(8, 'jaccard', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(2, 'jaro', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(3, 'jaro_winkler', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(4, 'jaro_winkler', rbaseTerms, rmismatchTerms, rspecialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(11, 'monge_elkan', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(7, 'cosine', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(9, 'strike_a_match', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(12, 'soft_jaccard', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        # tot_res += self._generic_evaluator(5, 'sorted_winkler', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(10, 'skipgram', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(13, 'davies', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(14, 'l_jaro_winkler', baseTerms, mismatchTerms, specialTerms, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(15, 'l_jaro_winkler', rbaseTerms, rmismatchTerms, rspecialTerms, flag_true_match, custom_thres)


"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
# def damerau_levenshtein_distance(s1, s2):
#     d = {}
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     for i in xrange(-1, lenstr1 + 1):
#         d[(i, -1)] = i + 1
#     for j in xrange(-1, lenstr2 + 1):
#         d[(-1, j)] = j + 1
#
#     for i in xrange(lenstr1):
#         for j in xrange(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             d[(i, j)] = min(
#                 d[(i - 1, j)] + 1,  # deletion
#                 d[(i, j - 1)] + 1,  # insertion
#                 d[(i - 1, j - 1)] + cost,  # substitution
#             )
#             if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
#                 d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
#
#     return d[lenstr1 - 1, lenstr2 - 1]
