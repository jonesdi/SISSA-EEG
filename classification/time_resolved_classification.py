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

def classify(data, test_splits):

    results_list = list()
    ### Running on test split for each time point
    for s in test_splits:
    
        train_samples = list()
        train_labels = list()
        test_samples = list()
        test_labels = list()
        for cat_vecs_i, cat_vecs in enumerate(data.items()):
            current_s = s[cat_vecs_i]
            for vec_i, vec in enumerate(cat_vecs[1]):
                if vec_i not in current_s:
                    '''
                    ### Time-resolved
                    if len(vec.shape) > 1:
                        train_samples.append(vec[:, time_i])
                    ### Searchlight
                    else:
                        train_samples.append(vec)
                    '''
                    train_samples.append(vec)

                    train_labels.append(cat_vecs[0])
                else:
                    '''
                    ### Time-resolved
                    if len(vec.shape) > 1:
                        test_samples.append(vec[:, time_i])
                    ### Searchlight
                    else:
                        test_samples.append(vec)
                    '''
                    test_samples.append(vec)
                    test_labels.append(cat_vecs[0])

        svm_model = svm.SVC().fit(train_samples, train_labels)
        accuracy = svm_model.score(test_samples, test_labels)
        results_list.append(accuracy)

    ### Averaging performances at a given time
    average_score = numpy.average(results_list)-.5
    #median_score = numpy.median(results_list)
            
    return average_score

def run_searchlight_classification(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    eeg = all_args[3]
    cluster = all_args[4]
    test_splits = all_args[5]

    places = list(cluster[0])
    start_time = cluster[1]

    eeg_scores = list()

    ### Reducing eeg to the relevant cluster
    eeg = {k : [vec[places, start_time:start_time+16].flatten() for vec in v] for k, v in eeg.items()}

    accuracy_score = classify(eeg, test_splits)

    return [(places[0], start_time), accuracy_score]

def run_classification(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    general_output_folder = all_args[3]
    
    eeg = SubjectData(exp, n, args)
    data = eeg.eeg_data
    times = list(eeg.times)

    for awareness, vecs in data.items():    

        test_splits = eeg.permutations[awareness]

        ### Time-resolved classification

        ### Classifying each time point
        scores_times = list()
        for time_i, time in tqdm(enumerate(times)):
            time_t_data = {k : [vec[:, time_i] for vec in v] for k, v in vecs.items()}
            scores_times.append(classify(time_t_data, test_splits))
            del time_t_data

        ### Writing to file
        with open(os.path.join(general_output_folder, \
                     'sub_{:02}_{}_scores.txt'.\
                     format(n+1, awareness)), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in scores_times:
                o.write('{}\t'.format(d))
