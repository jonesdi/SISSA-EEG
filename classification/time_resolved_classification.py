import collections
import itertools
import argparse
import os
import scipy
import random
import mne
import numpy
import scipy
import sklearn

from sklearn import svm
from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm
from io_utils import ExperimentInfo, SubjectData

def run_classification(exp, eeg, n, args):

    cats = list(exp.cats_to_words.keys())

    data = eeg.eeg_data

    folds = 8000

    out_path = os.path.join('classification_per_subject', \
                            'sub-{:02}'.format(n+1))
    os.makedirs(out_path, exist_ok=True)

    for awareness, vecs in data.items():    

        current_data = {c : list() for c in cats}
        for w, vec in vecs.items():
            current_data[exp.words_to_cats[w]].append(vec)

        total_number_words = min([len(v) for k, v in current_data.items()])
        if total_number_words >= 5: ### Employing conditions with at least 10 words
            current_data = {k : v[:total_number_words] for k, v in current_data.items()}
            number_test_samples = max(1, int(0.1 * total_number_words))
            word_splits = list(itertools.combinations(list(range(total_number_words)), number_test_samples))
            test_splits = list(itertools.product(word_splits, repeat=2))

            scores_times = list()
            for time_i, time in tqdm(enumerate(list(eeg.times))):
                time_list = list()
                for s in test_splits[:folds]:
                
                    train_samples = list()
                    train_labels = list()
                    test_samples = list()
                    test_labels = list()
                    for cat_vecs_i, cat_vecs in enumerate(current_data.items()):
                        current_s = s[cat_vecs_i]
                        for vec_i, vec in enumerate(cat_vecs[1]):
                            if vec_i not in current_s:
                                train_samples.append(vec[:, time_i])
                                train_labels.append(cat_vecs[0])
                            else:
                                test_samples.append(vec[:, time_i])
                                test_labels.append(cat_vecs[0])

                    svm_model = svm.SVC().fit(train_samples, train_labels)
                    accuracy = svm_model.score(test_samples, test_labels)
                    time_list.append(accuracy)
                scores_times.append(numpy.average(time_list))

            with open(os.path.join(out_path, 'sub_{:02}_{}_scores.txt'.format(n+1, awareness)), 'w') as o:
                for t in eeg.times:
                    o.write('{}\t'.format(t))
                o.write('\n')
                for d in scores_times:
                    o.write('{}\t'.format(d))
        else:
            print('not enough words for: sub {}, condition {}'.format(n+1, awareness))
