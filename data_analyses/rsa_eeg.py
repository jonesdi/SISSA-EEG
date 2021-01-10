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

from io_utils import ComputationalModels, EvokedResponses, SearchlightClusters

def restrict_evoked_responses(args, evoked_responses):


    return restricted_evoked_responses

parser = argparse.ArgumentParser()
parser.add_argument('--targets_only', action='store_true', default=False, help='Indicates whether to only use target words or all words')
parser.add_argument('--accuracy_analysis', action='store_true', default=True, help='Decides whether to do basic level analyses or higher level analyses')
parser.add_argument('--permutation', action='store_true', default=False, help='Indicates whether to run a permutation analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'target_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_models', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

event_mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}
if args.computational_model == 'w2v':
    computational_model = ComputationalModels().w2v
searchlight_clusters = SearchlightClusters()

hop = 2
temporal_window_size = 4

### RSA

all_results = collections.defaultdic(dict)

for s in range(3, 17): 

    evoked_responses = EvokedResponses(s)

    ### Selecting evoked responses for the current pairwise similarity computations
    
    selected_evoked = restrict_evoked_responses(args, evoked_responses)

    subject_results = collections.defaultdict(dict)

    print('\nNow computing and comparing similarities...')

    for condition, evoked_dict in selected_evoked.items():

        evoked_dicts = {k : numpy.average(v, axis=0) for k, v in evoked_dict.items()}

        word_combs = [k for k in itertools.combinations(evoked_dict.keys(), r=2)]

        computational_scores = collections.defaultdict(list)

        for word_one, word_two in word_combs:

            computational_scores.append(computational_model[word_one][word_two])

        current_condition_rho = collections.defaultdict(list)

        for center in range(128):

            relevant_electrode_indices = searchlight_clusters.neighbors[center] + [center]

            for t in range(0, len(evoked_responses.time_points), 2):

                eeg_similarities = list()

                relevant_time_indices = [t+i for i in range(4)]

                for word_one, word_two in word_combs:
                    for relevant_time in relevant_time_indices:
                        for relevant_electrode in relevant_electrode_indices:
                        
                            eeg_one.append(evoked_dicts[word_one][relevant_electrode, relevant_electrode])
                            eeg_two.append(evoked_dicts[word_two][relevant_electrode, relevant_electrode])

                            eeg_similarities.append(scipy.stats.spearmanr(eeg_one, eeg_two)[0])

                rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
                current_condition_rho[center].append(rho_score)

        subject_results[condition] = current_condition_rho

    all_results[s] = subject_results

