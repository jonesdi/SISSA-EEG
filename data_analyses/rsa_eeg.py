import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse

from tqdm import tqdm

from io_utils import ComputationalModels, EvokedResponses, SearchlightClusters

def restrict_evoked_responses(args, evoked_responses):

    events = evoked_responses.events
    epochs = evoked_responses.original_epochs
    assert len(events) == len(epochs)

    restricted_evoked_responses = collections.defaultdict(lambda : collections.defaultdict(list))

    for current_epoch_index in range(len(events)):

        current_event = events[current_epoch_index]

        ### Considering only targets
        if args.word_selection == 'target_only':
            if current_event[1] == 'target':
                ### Creating 2 conditions: correct/wrong
                if args.analysis == 'objective_accuracy':
                    restricted_evoked_responses[current_event[2]][current_event[0]].append(epochs[current_epoch_index])
                ### Creating 3 conditions: 1/2/3
                elif args.analysis == 'subjective_judgments':
                    restricted_evoked_responses[current_event[2]][current_event[3]].append(epochs[current_epoch_index])

        ### Considering all words
        else:
            ### Creating 2 conditions: correct/wrong
            if args.analysis == 'objective_accuracy':
                restricted_evoked_responses[current_event[2]][current_event[0]].append(epochs[current_epoch_index])
            ### Creating 3 conditions: 1/2/3
            elif args.analysis == 'subjective_judgments':
                restricted_evoked_responses[current_event[2]][current_event[3]].append(epochs[current_epoch_index])

    return restricted_evoked_responses

parser = argparse.ArgumentParser()
parser.add_argument('--permutation', action='store_true', default=False, help='Indicates whether to run a permutation analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'target_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_models', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

if args.computational_models == 'w2v':
    computational_model = ComputationalModels().w2v
searchlight_clusters = SearchlightClusters()

base_folder = os.path.join('rsa_maps', 'rsa_{}_{}'.format(args.analysis, args.word_selection, args.computational_models))
os.makedirs(os.path.join(base_folder))

hop = 2
temporal_window_size = 4

### RSA

all_results = collections.defaultdict(dict)

for s in range(3, 17): 

    evoked_responses = EvokedResponses(s)

    ### Selecting evoked responses for the current pairwise similarity computations
    
    selected_evoked = restrict_evoked_responses(args, evoked_responses)

    subject_results = collections.defaultdict(dict)
    subject_info = collections.defaultdict(lambda : collections.defaultdict(list))

    print('\nNow computing and comparing similarities for subject {}...'.format(s))

    for condition, evoked_dict in selected_evoked.items():

        print('Current condition: {}'.format(condition))

        evoked_dicts = {k : numpy.average(v, axis=0) for k, v in evoked_dict.items()}

        present_words = [k for k in evoked_dict.keys()]

        subject_info[condition]['words used'] = present_words

        word_combs = [k for k in itertools.combinations(present_words, r=2)]

        computational_scores = list()

        for word_one, word_two in word_combs:

            computational_scores.append(computational_model[word_one][word_two])

        current_condition_rho = collections.defaultdict(list)

        for center in tqdm(range(128)):

            relevant_electrode_indices = searchlight_clusters.neighbors[center]

            for t in range(0, len(evoked_responses.time_points), 2):

                eeg_similarities = list()

                relevant_time_indices = [t+i for i in range(4)]

                for word_one, word_two in word_combs:

                    eeg_one = list()
                    eeg_two = list()

                    for relevant_time in relevant_time_indices:
                        for relevant_electrode in relevant_electrode_indices:
                        
                            eeg_one.append(evoked_dicts[word_one][relevant_electrode, relevant_time])
                            eeg_two.append(evoked_dicts[word_two][relevant_electrode, relevant_time])

                    eeg_similarities.append(scipy.stats.spearmanr(eeg_one, eeg_two)[0])

                rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
                current_condition_rho[center].append(rho_score)

        subject_results[condition] = current_condition_rho

    ### Writing to file
    for s, condition_dict in subject_results.items():
        subject_folder = os.path.join(base_folder, 'sub-{:02}'.format(s))
        os.makedirs(subject_folder, exist_ok=True)
        for condition, clusters_dict in condition_dict.items():

            ### Writing the Spearman rho maps
            with open(os.path.join(subject_folder, '{}.map'.format(condition)), 'w') as o:
                o.write('Searchlight cluster index\tSpearman Rho per time window\n')
                for cluster_index, rho_map in clusters_dict.items():
                    o.write('{}\t'.format(cluster_index))
                    for rho in rho_map:
                        o.write('{}\t'.format(rho))
                    o.write('\n')

        ### Writing the words actually used
        with open(os.path.join(subject_folder, 'words_used_info.txt'), 'w') as o:
            for condition, words_used in subject_info.items():
                o.write('Condition:\t{}\nNumber of words used:\t{}\n\n'.format(condition, len(words_used)))
