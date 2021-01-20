import argparse
import os
import collections
import numpy
from scipy import stats

from searchlight import SearchlightClusters
from io_utils import EvokedResponses
from plot_utils import basic_line_plot_all_electrodes_subject_p_values, basic_scatter_plot_all_electrodes_subject_p_values, \
                       line_and_scatter_plot_all_electrodes_subject_p_values 

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--searchlight', action='store_true', default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_model', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

if args.analysis == 'objective_accuracy':
    conditions = ['correct', 'wrong']
elif args.analysis == 'subjective_judgments':
    conditions = ['low', 'medium', 'high']

timepoint_converter = EvokedResponses(3).time_points
final_plot = collections.defaultdict(list)

for s in tqdm(range(3, 17)):
    ### Collecting the true results
    true_data_folder = os.path.join('rsa_maps', 'rsa_searchlight_{}_{}_{}_{}_permutation_False'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{:02}'.format(s))

    results = collections.defaultdict(lambda : collections.defaultdict(list))
    counts = collections.defaultdict(int)
    for condition in conditions:
        try:
            with open(os.path.join(true_data_folder, '{}.map'.format(condition)), 'r') as input_file:
                all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1]
            results[condition]['true'] = all_electrodes
        except FileNotFoundError:
            conditions = [c for c in conditions if c != condition]

        try:
            with open(os.path.join(true_data_folder, 'words_used_info.txt'), 'r') as input_file:
                word_counts = [l.strip().split('\t')[1] for l in input_file.readlines() if len(l.split('\t'))>1]
            counts[word_counts[0]] = int(word_counts[1])
            counts[word_counts[2]] = int(word_counts[3])
        except FileNotFoundError:
            counts['correct'] = 'unk'
            counts['wrong'] = 'unk'

    for condition in conditions:
        for i in range(1, 301):
            permutation_data_folder = os.path.join('rsa_maps', 'rsa_searchlight_{}_{}_{}_{}_permutation_True'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{}_{:03}'.format(s, i))
            try:
                with open(os.path.join(permutation_data_folder, '{}.map'.format(condition)), 'r') as input_file:
                    all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1]
                results[condition][i] = all_electrodes
            except FileNotFoundError:
                print(i)

    plot_path = os.path.join('p-value_plots', 'rsa_searchlight_{}_{}_{}_{}'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{:02}'.format(s))
    os.makedirs(plot_path, exist_ok=True)
    #t_values = collections.defaultdict(lambda: collections.defaultdict(list))
    p_values = collections.defaultdict(lambda: collections.defaultdict(list))
    original_rhos = collections.defaultdict(list)
    permutation_rhos = collections.defaultdict(list)

    for condition, condition_dict in results.items():
        with open(os.path.join(plot_path, '{}_significant_time_points.txt'.format(condition)), 'w') as o:
            o.write('Condition\ttime point\tp-value\tt-value\n')
        time_points = [t for t in range(len(condition_dict['true']))]

        for t in time_points:

            readable_time_point = timepoint_converter[t]
            rho_at_t = float(condition_dict['true'][t])
            rho_scores = [float(v[t]) for k, v in condition_dict.items() if k!='true']
            p_value = 1.-(stats.percentileofscore(rho_scores, rho_at_t)/100.)
            t_value = stats.t.ppf(1.-(p_value), 299)

            if p_value <= .05:
                #t_values[condition]['all_electrodes'].append((readable_time_point, t_value))
                with open(os.path.join(plot_path, '{}_significant_time_points.txt'.format(condition)), 'a') as o:
                    o.write('{}\t{}\t{}\t{}\n'.format(condition, readable_time_point, p_value, t_value))

            original_rhos[condition].append(rho_at_t)
            permutation_rhos[condition].append(rho_scores)
            p_values[condition]['all_electrodes'].append((readable_time_point, p_value))

        line_and_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], p_values, original_rhos, permutation_rhos, condition, plot_path, counts)

        final_plot[condition].append(p_values[condition]['all_electrodes'])
    #basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], t_values, 't', plot_path)
    basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], p_values, 'p', plot_path, counts)

#final_plot = {k : numpy.average(v, axis=0) for k, v in final_plot.items()}
plot_path = os.path.join('p-value_plots', 'rsa_searchlight_{}_{}_{}_{}'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model))
basic_scatter_plot_all_electrodes_subject_p_values('all_subjects', [timepoint_converter[t] for t in time_points], final_plot, 'average_p', plot_path)
