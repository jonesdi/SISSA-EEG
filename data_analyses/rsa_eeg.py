import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from io_utils import ComputationalModels, EvokedResponses

parser = argparse.ArgumentParser()
parser.add_argument('--targets_only', action='store_true', default=False, help='Indicates whether to only use target words or all words')
parser.add_argument('--accuracy_analysis', action='store_true', default=True, help='Decides whether to do basic level analyses or higher level analyses')
parser.add_argument('--permutation', action='store_true', default=False, help='Indicates whether to run a permutation analysis or not')
args = parser.parse_args()

event_mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}
computational_models = ComputationalModels()

### RSA

for s in range(3, 17): 

    evoked_responses = EvokedResponses(3)

    subject_results = collections.defaultdict(list)

    print('\nNow computing and comparing similarities...')

    for certainty_level, indices in events_selector.items():
        current_evoked_eeg = collections.defaultdict(list)
        for index in indices:
            current_event = events[index] 
            current_word = current_event[0]
            current_evoked_eeg[current_word].append(epochs[index])
        current_evoked_eeg = {k : numpy.average(v, axis=0) for k, v in current_evoked_eeg.items()}

        word_combs = [k for k in itertools.combinations(current_evoked_eeg.keys(), r=2)]

        eeg_similarities = collections.defaultdict(list)
        computational_models_ordered = collections.defaultdict(list)
        for word_one, word_two in word_combs:

            for key, model in computational_models.items():
                if (word_one, word_two) in model.keys():
                    current_key = (word_one, word_two)
                elif (word_two, word_one) in model.keys():
                    current_key = (word_two, word_one)
                else:
                    print('A couple of words were absent from the model: {} and {}'.format(word_one, word_two))

                computational_models_ordered[key].append(model[current_key])

            for sample_point, time_point in time_points.items():
                eeg_one = current_evoked_eeg[word_one][:, sample_point]
                eeg_two = current_evoked_eeg[word_two][:, sample_point]
                eeg_similarities[time_point].append(scipy.stats.pearsonr(eeg_one, eeg_two)[0])

        for time_point, eeg_time_point in eeg_similarities.items():
            correlation_with_computational = list()
            for key, model in computational_models_ordered.items():
                score = scipy.stats.pearsonr(eeg_time_point, model)[0]
                correlation_with_computational.append((key, score))
            subject_results[certainty_level].append(correlation_with_computational)
        subject_results[certainty_level].append('n_words={}'.format(len(current_evoked_eeg.keys())))

