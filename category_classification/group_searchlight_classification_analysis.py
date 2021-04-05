import argparse
import os
import collections
import numpy
import re
import logging
import itertools
import functools
import mne
import pickle
import scipy
import multiprocessing

from scipy import stats
from matplotlib import pyplot
from tqdm import tqdm

import sys
sys.path.append('../rsa_analyses')
from searchlight import SearchlightClusters
from io_utils import EvokedResponses
from plot_utils import plot_two, plot_three, confusion_matrix
from rsa_utils import prepare_folder

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--searchlight', action='store_true', default=True, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='both_worlds', choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_model', default='w2v', choices=['cslb', 'CORnet', 'visual', 'orthography', 'w2v', 'original_cooc', 'ppmi', 'new_cooc', 'wordnet'], help='Indicates which similarities to use for comparison to the eeg similarities')
parser.add_argument('--comparisons_correction', default='fdr', choices=['fdr', 'cluster', 'cluster_tfce', 'maximal_permutation'], help='Indicates which multiple comparisons correction approach to use')
parser.add_argument('--hop', default=3, type=int, help='Indicates which similarities to use for comparison to the eeg similarities')
parser.add_argument('--temporal_window_size', default=7, type=int, help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

if args.analysis == 'objective_accuracy':
    conditions = ['correct', 'wrong']
elif args.analysis == 'subjective_judgments':
    conditions = ['low', 'medium', 'high']
elif args.analysis == 'both_worlds':
    conditions = ['aware', 'unaware']

electrode_index_to_code = SearchlightClusters().index_to_code
mne_adj_matrix = SearchlightClusters().mne_adjacency_matrix
timepoint_converter = EvokedResponses(3).time_points
if args.searchlight:
    searchlight_converter = {i : v for i, v in enumerate([t for t in range(0, len(timepoint_converter), args.hop)])}
chosen_timepoints = [k for k, v in searchlight_converter.items() if timepoint_converter[v]>=-.2 and timepoint_converter[v]<1. and k<80]

plot_path = os.path.join('group_searchlight_classification_plots', args.analysis, args.word_selection)
os.makedirs(plot_path, exist_ok=True)

final_plot = collections.defaultdict(list)
true_data = collections.defaultdict(lambda: collections.defaultdict(lambda : collections.defaultdict(list)))


if 'maximal' not in args.comparisons_correction:

    for condition in conditions:

        condition_list = list()

        for s in tqdm(range(2, 17)):

            ### Collecting the true results
            base_folder = os.path.join('classification_maps', args.analysis, \
                                      'window_{}_hop_{}'.format(args.temporal_window_size, args.hop), \
                                      args.word_selection, 'sub-{:02}'.format(s))

            sub_list = list()    

            try:
                with open(os.path.join(base_folder, '{}.map'.format(condition)), 'r') as input_file:
                    all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1:]
                if len(all_electrodes) > 1:
                    #max_time_index = len(all_electrodes[0])
                    #for t in range(max_time_index):
                    for t in chosen_timepoints:
                        if 'cluster'in args.comparisons_correction:
                            t_list = [float(elec[t])-.5 for elec in all_electrodes]
                        else:
                            t_list = [float(elec[t]) for elec in all_electrodes]
                        sub_list.append(t_list)
            except FileNotFoundError:
                pass

            if len(set([str(k) for t in sub_list for k in t])) != 1 and len(sub_list) >0:
                condition_list.append(sub_list)

        condition_array = numpy.array(condition_list)

        if args.comparisons_correction == 'cluster':
            res = mne.stats.spatio_temporal_cluster_1samp_test(condition_array, tail=1, adjacency=mne_adj_matrix, max_step=1, n_jobs=os.cpu_count()-1)
            significant_points = res[2].reshape(res[0].shape).T

        if args.comparisons_correction == 'cluster_tfce':
            res = mne.stats.spatio_temporal_cluster_1samp_test(condition_array, tail=1, adjacency=mne_adj_matrix, threshold=dict(start=0, step=0.2), n_jobs=os.cpu_count()-1, n_permutations=1024)
            significant_points = res[2].reshape(res[0].shape).T

        '''
        ps = [(i, k) for i, k in enumerate(res[2])]
        highest_p = [k for k in sorted(ps, key=lambda item: item[1]) if k[1] <= .05]
        times = [res[1][i[0]][0][0] for i in highest_p]
        readable_times = [timepoint_converter[searchlight_converter[chosen_timepoints[k]]] for k in times]
        places= [res[1][i[0]][1][0] for i in highest_p]

        readable_rhos = list()
        for i in range(len(places)):
            place = places[i]
            index_rhos = list()
            for s in range(len(condition_list)):
                original_t = chosen_timepoints[times[i]]
                original_rho = condition_array[s, original_t, place]
                index_rhos.append(original_rho)
            readable_rhos.append(index_rhos)
        '''

        ### FDR

        if args.comparisons_correction == 'fdr':
        
            fdr_array = condition_array.reshape(condition_array.shape[0], -1)
            #sum_array = numpy.full_like(fdr_array, fill_value=-.5)
            #fdr_array = numpy.sum(fdr_array, sum_array)
            p_values = list()
            for variable in range(fdr_array.shape[-1]):
                #p_value =mne.stats.permutation_t_test(fdr_array[:, variable], tail=1, n_jobs=os.cpu_count()-1)
                p_value = stats.ttest_1samp(fdr_array[:, variable], popmean=.5, alternative='greater')[1]
            #p_values = mne.stats.permutation_t_test(fdr_array, tail=1, n_jobs=os.cpu_count()-1)[1]

                p_values.append(p_value)

            sign_mask, adjusted_p_values = mne.stats.fdr_correction(p_values)

            sign_mask = sign_mask.reshape(condition_array.shape[-2:])
            significant_points = adjusted_p_values.reshape(condition_array.shape[-2:]).T
             

        ### Plotting the results

        significant_points = -numpy.log(significant_points)
        significant_points[significant_points<=-numpy.log(0.05)] = 0.0

        time_indices=[i[0] for i in enumerate(numpy.nansum(significant_points.T, axis=1)>0.) if i[1]==True]
        tmin=timepoint_converter[searchlight_converter[chosen_timepoints[0]]]
        info = mne.create_info(ch_names=[v for k, v in SearchlightClusters().index_to_code.items()], sfreq=204.8/3, ch_types='eeg')
        evoked = mne.EvokedArray(significant_points, info=info,tmin=tmin)
        
        montage=mne.channels.make_standard_montage('biosemi128')
        evoked.set_montage(montage)

        for i in range(2):

            mode = 'all' if i==0 else 'significant'

            title='{} time points for model: {} - Condition: {}'.format(mode.capitalize(), args.computational_model, condition)

            if mode == 'significant':
                if len(time_indices) >= 1:
                    evoked.plot_topomap(ch_type='eeg', time_unit='s', times=[evoked.times[i] for i in time_indices], units='-log(p)\nif\np<=.05', ncols=12, nrows='auto', vmin=0., scalings={'eeg':1.}, cmap='PuBu', title=title)
            else:
                
                evoked.plot_topomap(ch_type='eeg', time_unit='s', times=[i for i in evoked.times], units='-log(p)\nif\np<=.05', ncols=12, nrows='auto', vmin=0., scalings={'eeg':1.}, cmap='PuBu', title=title)

            pyplot.savefig(os.path.join(plot_path, '{}_{}.png'.format(mode, condition)), dpi=600)
            pyplot.clf()

else:

    logging.info('Now loading the data and permuting it...')

    ### Collecting the true results
    for s in tqdm(range(2, 17)):

        
        for condition in conditions:
            
            base_folder = prepare_folder(args, s)

            try:
                with open(os.path.join(base_folder, '{}.map'.format(condition)), 'r') as input_file:
                    all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1:]
                if len(all_electrodes) > 1:
                    for elec_index, electrode_time_points in enumerate(all_electrodes):
                        elec_code = electrode_index_to_code[elec_index]

                        true_values = [float(n) for n in electrode_time_points]
                        for t, true_value in enumerate(true_values):
                            time_point = timepoint_converter[searchlight_converter[t]]
                            true_data[condition][time_point][elec_code].append(true_value)

            except FileNotFoundError:
                pass

    '''
    logging.info('Now running the analyses...')

    if args.comparisons_correction == 'fdr':

        results = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(float)))
        for condition, condition_dict in true_data.items():
            fdr = list()
            #maximal_distribution = perm_dict[condition]
            for time_point, time_dict in condition_dict.items():
                for elec_code, elec_list in time_dict.items():
                    #elec_score = scipy.stats.ttest_1samp(elec_list, alternative='greater', popmean=0.0, nan_policy='omit')[1]
                    elec_score = scipy.stats.ttest_1samp(elec_list, alternative='greater', popmean=0.0)[1]
                    #p_value = 1.-(stats.percentileofscore(maximal_distribution, elec_score)/100.)
                    #if p_value/2 <= .05:
                        #print([elec_code, time_point])
                    #results[condition][elec_code][time_point] = p_value
                    fdr.append((elec_score, (elec_code, time_point)))
            res = mne.stats.fdr_correction([k[0] for k in fdr])
            sig = [k for k in res[0] if str(k) != 'False']
            logging.info('significant points for condition {}: {}'.format(condition, sig))
    '''
