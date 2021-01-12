import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse
from multiprocessing import Process

from tqdm import tqdm

from io_utils import ComputationalModels, EvokedResponses, SearchlightClusters
from plot_utils import basic_line_plot_searchlight_electrodes

def restrict_evoked_responses(args, evoked_responses):

    events = evoked_responses.events
    if args.permutation:
        event_keys = [k for k in events.keys()]
        random_event_values = random.sample([v for k, v in events.items()], k=len(event_keys))
        events = {k : v for k, v in zip(event_keys, random_event_values)}
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

def run_searchlight(evoked_dict, word_combs, computational_scores, time_points): 

    ### Loading the searchlight clusters
    searchlight_clusters = SearchlightClusters()
    hop = 2
    temporal_window_size = 4

    current_condition_rho = collections.defaultdict(list)


    for center in tqdm(range(128)):

        relevant_electrode_indices = searchlight_clusters.neighbors[center]

        for t in time_points:

            eeg_similarities = list()

            relevant_time_indices = [t+i for i in range(temporal_window_size)]

            for word_one, word_two in word_combs:

                eeg_one = list()
                eeg_two = list()

                for relevant_time in relevant_time_indices:
                    for relevant_electrode in relevant_electrode_indices:
                    
                        eeg_one.append(evoked_dict[word_one][relevant_electrode, relevant_time])
                        eeg_two.append(evoked_dict[word_two][relevant_electrode, relevant_time])

                word_comb_score = scipy.stats.spearmanr(eeg_one, eeg_two)[0]
                eeg_similarities.append(word_comb_score)

            rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
            current_condition_rho[center].append(rho_score)

    return current_condition_rho

def run_all_electrodes_rsa(evoked_dict, word_combs, computational_scores, time_points): 

    current_condition_rho = collections.defaultdict(list)

    for relevant_time in tqdm(time_points):

        eeg_similarities = list()

        for word_one, word_two in word_combs:

            eeg_one = evoked_dict[word_one][:, relevant_time]
            eeg_two = evoked_dict[word_two][:, relevant_time]

            word_comb_score = scipy.stats.spearmanr(eeg_one, eeg_two)[0]
            eeg_similarities.append(word_comb_score)

        rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
        current_condition_rho['all electrodes'].append(rho_score)

    return current_condition_rho

def run_rsa(args, s, selected_evoked, computational_model, all_time_points):

    base_folder = os.path.join('rsa_maps', 'rsa_searchlight_{}_{}_{}_{}_permutation_{}'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model, True if args.permutation else False))
    os.makedirs(os.path.join(base_folder), exist_ok=True)

    subject_results = collections.defaultdict(dict)
    subject_info = collections.defaultdict(lambda : collections.defaultdict(list))

    for condition, evoked_dict in selected_evoked.items():

        #print('Current condition: {}'.format(condition))

        evoked_dict = {k : numpy.average(v, axis=0) for k, v in evoked_dict.items()}

        present_words = [k for k in evoked_dict.keys()]

        subject_info[condition]['words used'] = present_words

        word_combs = [k for k in itertools.combinations(present_words, r=2)]

        computational_scores = list()

        for word_one, word_two in word_combs:

            computational_scores.append(computational_model[word_one][word_two])

        ### Differentiating among searchlight and full-cap RSA
        if args.searchlight:
            time_points = [t for t in range(0, len(all_time_points), hop)]
            current_condition_rho = run_searchlight(evoked_dict, word_combs, computational_scores, time_points) 
        else:
            time_points = [k for k in all_time_points.keys()]
            current_condition_rho = run_all_electrodes_rsa(evoked_dict, word_combs, computational_scores, time_points) 

        subject_results[condition] = current_condition_rho

    ### Writing to file
    for condition, condition_dict in subject_results.items():
        subject_folder = os.path.join(base_folder, 'sub-{:02}'.format(s))
        os.makedirs(subject_folder, exist_ok=True)

        ### Writing the Spearman rho maps
        with open(os.path.join(subject_folder, '{}.map'.format(condition)), 'w') as o:
            if args.searchlight:
                o.write('Searchlight cluster index\tSpearman Rho per time window\n')
            else:
                o.write('Electrode type\tSpearman Rho per time window\n')
            for cluster_index, rho_map in condition_dict.items():
                o.write('{}\t'.format(cluster_index))
                for rho in rho_map:
                    o.write('{}\t'.format(rho))
                o.write('\n')

        ### Plotting the basic plot for each condition
        basic_line_plot_searchlight_electrodes([all_time_points[k] for k in time_points], condition_dict, condition, args.computational_model, subject_info[condition]['words used'], subject_folder)

    ### Writing the words actually used
    with open(os.path.join(subject_folder, 'words_used_info.txt'), 'w') as o:
        for condition, condition_info in subject_info.items():
            words_used = condition_info['words used']
            o.write('Condition:\t{}\nNumber of words used:\t{}\n\n'.format(condition, len(words_used)))

def rsa_per_subject(args, s):

    evoked_responses = EvokedResponses(s)
    all_time_points = evoked_responses.time_points

    if args.permutation:
        for permutation in range(1, 301):

            ### Selecting evoked responses for the current pairwise similarity computations
            selected_evoked = restrict_evoked_responses(args, evoked_responses)
            run_rsa(args, '{}_{:03}'.format(s, permutation), selected_evoked, computational_model, all_time_points)

    else:
        ### Selecting evoked responses for the current pairwise similarity computations
        selected_evoked = restrict_evoked_responses(args, evoked_responses)
        run_rsa(args, s, selected_evoked, computational_model, all_time_points)


parser = argparse.ArgumentParser()
parser.add_argument('--permutation', action='store_true', default=False, help='Indicates whether to run a permutation analysis or not')
parser.add_argument('--searchlight', action='store_true', default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'target_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_model', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

### RSA

if __name__ == '__main__':
    
    ### Loading the computational model
    if args.computational_model == 'w2v':
        computational_model = ComputationalModels().w2v

    processes = list()
    for s in range(3, 17): 
        proc = Process(target=rsa_per_subject, args=(args, s,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
