# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import Counter
import json
import __main__

from femlAlgorithms import *
from helpers import normalize_str, getRelativePathtoWorking, StaticValues
from datasetcreator import filter_dataset, build_dataset_from_geonames
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.model_selection import train_test_split


class Evaluator:
    evaluatorType_action = {
        'SotAMetrics': calcSotAMetrics,
        'SotAML': calcSotAML,
        'customFEML': calcCustomFEML,
        'DLearning': calcDLearning,
        'TestMetrics': testMetrics,
        'customFEMLExtended': calcCustomFEMLExtended,
        'lSimilarityMetrics': calcLSimilarities,
        'customHyperparams': calcWithCustomHyperparams,
    }

    def __init__(self, ml_algs, sorting=False, stemming=False, canonical=False, permuted=False, only_latin=False, encoding=None):
        self.ml_algs = [x for x in ml_algs.split(',')]
        self.permuted = permuted
        self.stemming = stemming
        self.canonical = canonical
        self.sorting = sorting
        self.latin = only_latin
        self.encoding = encoding

        self.termsperalphabet = {}
        self.stop_words = []
        self.abbr = {'A': [], 'B': []}
        self.evalClass = None

    def initialize(self, dataset, evalType='SotAMetrics', njobs=2, accuracyresults=False):
        try:
            self.evalClass = self.evaluatorType_action[evalType](njobs, accuracyresults)
        except KeyError:
            print("Unkown method")
            return 1

        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                self.evalClass.preprocessing(row)

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt', feature_selection=None, features=None):
        if self.evalClass is not None:
            self.evalClass.freq_terms_list(self.encoding)

            lFeatures = [(True if x == 'True' else False) for x in features.split(',')] if feature_selection is None and features is not None else features
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                thres_type = 'orig'
                if self.sorting:
                    thres_type = 'sorted'
                if self.latin:
                    # thres_type += '_onlylatin'
                    thres_type += '_latin_EU/NA'
                if self.encoding:
                    thres_type += '_all'

                for row in reader:
                    self.evalClass.evaluate(
                        row, self.sorting, self.stemming, self.canonical, self.permuted, thres_type, lFeatures
                    )
            if hasattr(self.evalClass, "load_test_dataset"):
                self.evalClass.reset()
                with open(os.path.join(os.path.abspath(os.path.dirname(__main__.__file__)), config.test_dataset)) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')

                    thres_type = 'orig'
                    if self.sorting:
                        thres_type = 'sorted'
                    if self.latin:
                        # thres_type += '_onlylatin'
                        thres_type += '_latin_EU/NA'
                    if self.encoding:
                        thres_type += '_all'

                    for row in reader:
                        self.evalClass.load_test_dataset(
                            row, self.sorting, self.stemming, self.canonical, self.permuted, thres_type
                        )
            if hasattr(self.evalClass, "train_classifiers"):
                self.evalClass.train_classifiers(self.ml_algs, polynomial=False, standardize=True, fs_method=feature_selection, features=lFeatures)
            self.evalClass.print_stats()

    def evaluate_metrics_with_various_thres(self, dataset='dataset-string-similarity.txt', features='basic'):
        if self.evalClass is not None:
            self.evalClass.freq_terms_list(self.encoding)

            start_time = time.time()
            print("Reading dataset...")

            start_pos = 0
            end_pos = 12
            range_pos = -1
            if features == 'sorted':
                start_pos = 12
                end_pos = 25
            elif features == 'lgm':
                start_pos = 25
                end_pos = 38
                range_pos = 8

            fields = ["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"]
            X = pd.read_csv(dataset, sep='\t', header=None, names=fields)
            y = X['res']
            X.drop(columns=['res', "c1", "c2", "a1", "a2", "cc1", "cc2"], inplace=True)

            res = {key: np.zeros(shape=(5, 8)) for key in StaticValues.featureColumns[start_pos:end_pos]}
            fold = 1
            outer_cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=13)
            feml = FEMLFeatures()
            for train_idx, test_idx in outer_cv.split(X, y):
                print('Fold {}...'.format(fold))

                for n in [3.34] + list(range(4, range_pos)):
                    weight_combs = [
                        tuple(float(x / 10.0) for x in seq)
                        for seq in itertools.product([1, 2, 3, 4, 5, 2.5, 3.33], repeat=2)
                        if sum(seq) == (10 - n)
                    ]

                    for w in weight_combs:
                        w = (float(n / 10.0),) + w
                        feml.update_weights(w)
                        print('Computing stats for weights ({})'.format(','.join(map(str, w))))
                        print('Computing stats for threshold', end='')

                        X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
                        fX = None
                        if features == 'basic':
                            fX = np.asarray(map(self._compute_basic_features, X_train['s1'], X_train['s2']))
                        elif features == 'sorted':
                            fX = np.asarray(map(self._compute_sorted_features, X_train['s1'], X_train['s2']))
                        elif features == 'lgm':
                            fX = np.asarray(map(self.compute_features, X_train['s1'], X_train['s2']))

                        tmp_res = {key: [] for key in StaticValues.featureColumns[start_pos:end_pos]}

                        separator = ''
                        for i in range(30, 91, 5):
                            print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                            sys.stdout.flush()
                            separator = ','

                            tmp_nd = fX >= float(i / 100.0)
                            for idx, name in enumerate(StaticValues.featureColumns[start_pos:end_pos], start=start_pos):
                                prec, rec, f1, _ = precision_recall_fscore_support(y_train, tmp_nd[:, idx], average='binary')
                                tmp_res[name].append([
                                    accuracy_score(y_train, tmp_nd[:, idx]), prec, rec, f1, float(i / 100.0)
                                ])

                        fX = None
                        if features == 'basic':
                            fX = pd.DataFrame(
                                map(self._compute_basic_features, X_test['s1'], X_test['s2']),
                                columns=StaticValues.featureColumns[0:end_pos]
                            )
                        elif features in ['sorted', 'lgm']:
                            fX = pd.DataFrame(
                                map(self._compute_sorted_features, X_test['s1'], X_test['s2']),
                                columns=StaticValues.featureColumns[0:42]
                            )

                        for key, val in tmp_res.items():
                            max_val = max(val, key=lambda x: x[0])
                            tmp_nd = fX[key] >= max_val[4]

                            acc = accuracy_score(y_test, tmp_nd)
                            if acc > res[key][(fold -1), 0]:
                                res[key][(fold - 1), 0] = acc
                                res[key][(fold - 1), 1:4] = precision_recall_fscore_support(y_test, tmp_nd, average='binary')[:3]
                                res[key][(fold - 1), 4] = max_val[4]
                                res[key][(fold - 1), 5:8] = w
                        print()

                fold += 1
                print()

            for key, val in res.items():
                max_val = max(val, key=lambda x: x[0])
                print('{}: {}'.format(key, list(max_val)))

    def _compute_basic_features(self, s1, s2):
        return self.compute_features(s1, s2, False)

    def _compute_sorted_features(self, s1, s2):
        return self.compute_features(s1, s2, True)

    def compute_features(self, s1, s2, sorted=True):
        f = []
        for status in list({False, sorted}):
            a, b = transform(s1, s2, sorting=status, canonical=status)

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not status: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if status:
                sim14 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim15 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])

            if status:
                f.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15])
            else:
                f.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        if sorted:
            a, b = transform(s1, s2, sorting=True, canonical=True)

            sim1 = self._compute_lgm_sim(a, b, 'damerau_levenshtein')
            sim2 = self._compute_lgm_sim(a, b, 'davies')
            sim3 = self._compute_lgm_sim(a, b, 'skipgram')
            sim4 = self._compute_lgm_sim(a, b, 'soft_jaccard')
            sim5 = self._compute_lgm_sim(a, b, 'strike_a_match')
            sim6 = self._compute_lgm_sim(a, b, 'cosine')
            sim7 = self._compute_lgm_sim(a, b, 'jaccard')
            sim8 = self._compute_lgm_sim(a, b, 'monge_elkan')
            sim9 = self._compute_lgm_sim(a, b, 'jaro_winkler')
            sim10 = self._compute_lgm_sim(a, b, 'jaro')
            sim11 = self._compute_lgm_sim(a, b, 'jaro_winkler_r')
            sim12 = self._compute_lgm_sim(a, b, 'l_jaro_winkler')
            sim13 = self._compute_lgm_sim(a, b, 'l_jaro_winkler_r')
            sim14, sim15, sim16 = self._compute_lgm_sim_base_scores(a, b, 'damerau_levenshtein')

            f.append(
                [sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15, sim16])

        f = list(chain.from_iterable(f))

        return f

    def _compute_lgm_sim(self, s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
            s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])

        if metric in ['jaro_winkler_r', 'lgm_jaro_winkler_r']:
            return weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                metric[:-2], True if w_type == 'avg' else False
            )
        else:
            return weighted_terms(baseTerms, mismatchTerms, specialTerms, metric, True if w_type == 'avg' else False)

    @staticmethod
    def _compute_lgm_sim_base_scores(s1, s2, metric, w_type='avg'):
        base_t, mis_t, special_t = lsimilarity_terms(s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])
        return score_per_term(base_t, mis_t, special_t, metric)

    def evaluate_sorting_with_various_thres(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            start_time = time.time()
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                separator = ''
                print('Computing stats for threshold', end='')

                thres_type = 'orig'
                if self.sorting:
                    thres_type = 'sorted'
                if self.latin:
                    # thres_type += '_onlylatin'
                    thres_type += '_latin_EU/NA'
                if self.encoding:
                    thres_type += '_all'

                all_res = {}
                for m in StaticValues.methods: all_res[m[0]] = []
                for i in range(55, 86, 5):
                    print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                    sys.stdout.flush()
                    separator = ','

                    csvfile.seek(0)
                    for row in reader:
                        # if self.latin and (row['a1'] != 'LATIN' or row['a2'] != 'LATIN'): continue

                        self.evalClass.evaluate_sorting(row, float(i / 100.0), thres_type, self.stemming, self.permuted)
                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                    tmp_res = self.evalClass.get_stats()

                    for key, val in tmp_res.items():
                        all_res[key].append([float(i / 100.0), val])

                    self.evalClass.reset_vars()

            print('\nThe process took {0:.2f} sec'.format(time.time() - start_time))
            for k, val in all_res.items():
                if len(val) == 0:
                    print('{0} is empty'.format(k))
                    continue

                print(k, max(val, key=lambda x: x[1][0]))

    def test_cases(self, dataset, test_case):
        if test_case - 1 == 0:
            print("Not implemented yet!!!")
        elif test_case - 1 == 1:
            if not os.path.exists("output"):
                os.makedirs("output")

            ngram_stats = {
                '2gram': Counter(), '3gram': Counter(), '4gram': Counter(),
                'gram_token': Counter(), '2gram_token': Counter(), '3gram_token': Counter()
            }
            abbr_stats = Counter()
            orig_strs = {}
            no_dashed_strs = 0

            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                feml = FEMLFeatures()
                for row in reader:
                    row['s1'], row['s2'] = transform(row['s1'], row['s2'], canonical=True)

                    for sstr in ['s1', 's2']:
                        # calc the number of abbr that exist
                        abbr_str = feml.containsAbbr(row[sstr])
                        if abbr_str != '-':
                            if abbr_str not in orig_strs.keys(): orig_strs[abbr_str] = []
                            abbr_stats[abbr_str] += 1
                            orig_strs[abbr_str].append(row[sstr])

                        # search for dashes in strings
                        no_dashed_strs += feml.containsDashConnected_words(row[sstr])

                        row[sstr] = transform_str(row[sstr], canonical=True)
                        ngram_tokens, _ = normalize_str(row[sstr], self.stop_words)

                        for term in ngram_tokens:
                            ngram_stats['gram_token'][term] += 1
                        for gram in list(itertools.chain.from_iterable(
                                [[ngram_tokens[i:i + n] for i in range(len(ngram_tokens) - (n - 1))]
                                 for n in [2, 3]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram_token'][' '.join(gram)] += 1
                            else:
                                ngram_stats['3gram_token'][' '.join(gram)] += 1

                        # ngrams chars
                        # ngrams = zip(*[''.join(strA_ngrams_tokens)[i:] for i in range(n) for n in [2, 3, 4]])
                        for gram in list(itertools.chain.from_iterable(
                                [[''.join(ngram_tokens)[i:i + n] for i in range(len(''.join(ngram_tokens)) - (n - 1))]
                                 for n in [2, 3, 4]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram'][gram] += 1
                            elif len(gram) == 3:
                                ngram_stats['3gram'][gram] += 1
                            elif len(gram) == 4:
                                ngram_stats['4gram'][gram] += 1

            print("Found {} dashed words in the dataset.".format(no_dashed_strs))

            with open("./output/abbr.csv", "w+") as f:
                f.write('abbr\tcount\tstr\n')
                for value, count in abbr_stats.most_common():
                    f.write("{0}\t{1}\t{2}\n".format(value, count, ','.join(orig_strs[value][:10])))

            for nm in ngram_stats.keys():
                with open("./output/{0}s.csv".format(nm), "w+") as f:
                    f.write('gram\tcount\n')
                    for value, count in ngram_stats[nm].most_common():
                        f.write("{}\t{}\n".format(value.encode('utf8'), count))
        elif test_case - 1 == 2:
            if self.evalClass is not None:
                self.evalClass.freq_terms_list(self.encoding)

                print("Reading dataset...")
                with open(dataset) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')

                    all_res = {}
                    for m in StaticValues.methods: all_res[m[0]] = []
                    feml = FEMLFeatures()
                    print('====================================================================================')
                    print("The averaged lSimilarity is being tested by default. To test the normal one, update ")
                    print("parameter 'averaged'=False under 'class testMetrics' in file 'femlAlgorithms.py'")
                    print('====================================================================================')
                    for n in [3.34] + list(range(4, 8)):
                        weight_combs = [
                            tuple(float(x/10.0) for x in seq)
                            for seq in itertools.product([1, 2, 3, 4, 5, 2.5, 3.33], repeat=2)
                            if sum(seq) == (10 - n)
                        ]

                        for w in weight_combs:
                            w = (float(n/10.0), ) + w
                            feml.update_weights(w)
                            print('Computing stats for weights ({})'.format(','.join(map(str, w))))
                            print('Computing stats for threshold', end='')

                            start_time = time.time()

                            tmp_res = {}
                            for m in StaticValues.methods: tmp_res[m[0]] = []
                            separator = ''
                            for i in range(35, 81, 5):
                                print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                                separator = ','
                                #  required for python before 3.3
                                sys.stdout.flush()

                                internal_separator = ' ['
                                for k in range(60, 81, 5):
                                    print('{0}{1}'.format(internal_separator, float(k / 100.0)), end='')
                                    internal_separator = ', '
                                    #  required for python before 3.3
                                    sys.stdout.flush()

                                    csvfile.seek(0)
                                    for row in reader:
                                        self.evalClass.evaluate(
                                            row, self.sorting, self.stemming, self.canonical, self.permuted,
                                            float(i / 100.0), float(k / 100.0)
                                        )
                                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                                    res = self.evalClass.get_stats()

                                    for key, val in res.items():
                                        tmp_res[key].append([float(i / 100.0), val, list(w), float(k / 100.0)])

                                    self.evalClass.reset_vars()

                                print(']', end='')

                            print('\nThe process for weight ({0}) took {1:.2f} sec'.format(','.join(map(str, w)), time.time() - start_time))
                            for k, val in tmp_res.items():
                                if len(val) == 0:
                                    continue

                                all_res[k].extend(val)
                                print(k, max(val, key=lambda x: x[1][0]))

                    print("\nFinal Results")
                    for k, val in all_res.items():
                        if len(val) == 0:
                            continue

                        print(k, max(val, key=lambda x: x[1][0]))
        elif test_case - 1 == 3:
            if self.evalClass is not None:
                self.evalClass.freq_terms_list()

                print("Reading dataset...")
                with open(dataset) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')
                    for row in reader:
                        self.evalClass.evaluate(
                            row, self.sorting, self.stemming, self.canonical, self.permuted, 'sorted'
                        )
                    self.evalClass.debug_stats()
        elif test_case - 1 == 4:
            self.evalClass.freq_terms_list(self.encoding)

            output_f = open("./output/lsimilarity_terms.csv", "w+")
            output_f.write("res\tstr1\tbase_s1\tmismatch_s1\tspecial_s1\tstr2\tbase_s2\tmismatch_s2\tspecial_s2\n")
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile,
                                        fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                for row in reader:
                    a, b = transform(row['s1'], row['s2'], sorting=True, canonical=True)
                    baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(a, b, 0.55)
                    output_f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        row['res'].upper(),
                        row['s1'], ','.join(baseTerms['a']).encode('utf8'),
                        ','.join(mismatchTerms['a']).encode('utf8'),
                        ','.join(specialTerms['a']).encode('utf8'),
                        row['s2'], ','.join(baseTerms['b']).encode('utf8'),
                        ','.join(mismatchTerms['b']).encode('utf8'),
                        ','.join(specialTerms['b']).encode('utf8')
                    ))

            output_f.close()
        else:
            print("Test #{} does not exist!!! Please choose a valid test to execute.".format(test_case))

    def print_false_posneg(self, datasets):
        if not os.path.exists("output"):
            os.makedirs("output")

        if len(datasets) == 2:
            reader = pd.read_csv((datasets[0]), sep='\t', names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])
            results = pd.read_csv(datasets[1], sep='\t', names=["res1", "res2"])

            print("No of rows for dataset: {0}".format(results.shape[0]/2))
            resultDf = pd.concat([results.iloc[results.shape[0]/2:], results.iloc[:results.shape[0]/2]], ignore_index=True)
            mismatches = pd.concat([reader, resultDf], axis=1)

            negDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is True) & (mismatches.res1 != mismatches.res2)
            ]
            negDf.to_csv('./output/false_negatives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
            posDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is False) & (mismatches.res1 != mismatches.res2)
            ]
            posDf.to_csv('./output/false_positives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
        elif len(datasets) == 3:
            reader = pd.read_csv(getRelativePathtoWorking(datasets[0]), sep='\t',
                                 names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])

            res1 = pd.read_csv(
                datasets[1], sep='\t',
                names=["res1"] + list(map(lambda x: "res1_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res1_transformed_s1", "res1_transformed_s2"]
            )
            res2 = pd.read_csv(
                datasets[2], sep='\t',
                names=["res2"] + list(map(lambda x: "res2_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res2_transformed_s1", "res2_transformed_s2"]
            )

            mismatches = pd.concat([reader, res1, res2], axis=1)
            mismatches = mismatches.sort_values(by=['res'], ascending=False)

            for metric_name in StaticValues.methods_as_saved:
                negDf = mismatches[
                    (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                    (mismatches.res1 == mismatches['res1_{0}'.format(metric_name)]) &
                    (mismatches.res2 != mismatches['res2_{0}'.format(metric_name)])
                    ]
                negDf.to_csv('./output/false_enhancedmetric_{0}.csv'.format(metric_name), sep='\t',
                             encoding='utf-8', columns=['s1', 's2', 'res', "res2_transformed_s1", "res2_transformed_s2"])

                tmpDf = mismatches[ mismatches.res1 != mismatches.res2 ]
                if not tmpDf.empty: print(tmpDf)
        else: print("Wrong number {0} of input datasets to cmp".format(len(datasets)))

    def build_dataset(self):
        build_dataset_from_geonames(only_latin=self.latin)
        filter_dataset()
