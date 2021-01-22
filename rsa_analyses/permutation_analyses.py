import argparse
import os
import collections
import numpy
from scipy import stats

from searchlight import SearchlightClusters
from io_utils import EvokedResponses
from plot_utils import basic_line_plot_all_electrodes_subject_p_values, basic_scatter_plot_all_electrodes_subject_p_values, \
                       line_and_scatter_plot_all_electrodes_subject_p_values, subject_electrodes_scatter_plot, \
                       confusion_matrix 

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

final_plot = collections.defaultdict(list)
electrode_index_to_code = SearchlightClusters().index_to_code
timepoint_converter = EvokedResponses(3).time_points
if args.searchlight:
    searchlight_converter = {i : v for i, v in enumerate([t for t in range(0, len(timepoint_converter), 2)])}


for s in tqdm(range(3, 17)):
    ### Collecting the true results
    true_data_folder = os.path.join('rsa_maps', 'rsa_searchlight_{}_{}_{}_{}_permutation_False'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{:02}'.format(s))

    results = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(list)))
    counts = collections.defaultdict(int)

    for condition in conditions:
        try:
            with open(os.path.join(true_data_folder, '{}.map'.format(condition)), 'r') as input_file:
                all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1:]
            if len(all_electrodes) == 1:
                results[condition]['all']['true'] = [float(n) for n in all_electrodes[0]]
            else:
                for i, electrode in enumerate(all_electrodes):
                    results[condition][electrode_index_to_code[i]]['true'] = [float(n) for n in electrode]
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
        for perm in range(1, 301):
            permutation_data_folder = os.path.join('rsa_maps', 'rsa_searchlight_{}_{}_{}_{}_permutation_True'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{:02}_{:03}'.format(s, perm))
            try:
                with open(os.path.join(permutation_data_folder, '{}.map'.format(condition)), 'r') as input_file:
                    all_electrodes = [l.strip().split('\t')[1:] for l in input_file.readlines()][1:]
                if len(all_electrodes) == 1:
                    results[condition]['all'][perm] = [float(n) for n in all_electrodes[0]]
                else:
                    for i, electrode in enumerate(all_electrodes):
                        results[condition][electrode_index_to_code[i]][perm] = [float(n) for n in electrode]
            except FileNotFoundError:
                print(perm)

    plot_path = os.path.join('p-value_plots', 'rsa_searchlight_{}_{}_{}_{}'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model), 'sub-{:02}'.format(s))
    os.makedirs(plot_path, exist_ok=True)
    #t_values = collections.defaultdict(lambda: collections.defaultdict(list))
    all_p_values = collections.defaultdict(lambda: collections.defaultdict(list))

    for condition, condition_dict in results.items():

        if args.searchlight:
            subject_electrodes_plot = collections.defaultdict(list)
            rho_matrix = collections.defaultdict(list)
            p_matrix = collections.defaultdict(list)

        for electrode_name, electrode_dict in condition_dict.items(): 
            with open(os.path.join(plot_path, '{}_{}_significant_time_points.txt'.format(electrode_name, condition)), 'w') as o:
                o.write('Condition\ttime point\tp-value\tt-value\n')
            time_points = [t for t in range(len(electrode_dict['true']))]
            if args.searchlight:
                plot_time_points = [timepoint_converter[searchlight_converter[t]] for t in time_points]
            else:
                plot_time_points = [timepoint_converter[t] for t in time_points]
            original_rhos = list()
            permutation_rhos = list()
            p_values = list()

            for t in time_points:

                if args.searchlight:
                    readable_time_point = timepoint_converter[searchlight_converter[t]]
                else:
                    readable_time_point = timepoint_converter[t]
                rho_at_t = float(electrode_dict['true'][t])
                rho_scores = [float(v[t]) for k, v in electrode_dict.items() if k!='true']
                p_value = 1.-(stats.percentileofscore(rho_scores, rho_at_t)/100.)
                t_value = stats.t.ppf(1.-(p_value), 299)

                if p_value <= .05:
                    #t_values[condition]['all_electrodes'].append((readable_time_point, t_value))
                    with open(os.path.join(plot_path, '{}_{}_significant_time_points.txt'.format(electrode_name, condition)), 'a') as o:
                        o.write('{}\t{}\t{}\t{}\n'.format(condition, readable_time_point, p_value, t_value))
                    if args.searchlight:
                        subject_electrodes_plot[electrode_name].append(readable_time_point)

                original_rhos.append(rho_at_t)
                permutation_rhos.append(rho_scores)
                p_values.append((readable_time_point, p_value))

            rho_matrix[electrode_name[:1]].append(original_rhos)
            p_matrix[electrode_name[:1]].append([p[1] for p in p_values])


            line_and_scatter_plot_all_electrodes_subject_p_values(s, plot_time_points, p_values, original_rhos, permutation_rhos, condition, plot_path, electrode_name, counts)

            all_p_values[condition][electrode_name] = p_values
            #final_plot[condition][electrode_name] = p_values

        ### Plotting the searchlight cluster results
        if args.searchlight:
            subject_electrodes_scatter_plot(s, plot_time_points, subject_electrodes_plot, condition, plot_path)
            for electrode_bundle, rho_lists in rho_matrix.items():
                confusion_matrix(s, 'bundle_{}_rho'.format(electrode_bundle), rho_lists, [k for k in condition_dict.keys()], plot_time_points, condition, plot_path)
            for electrode_bundle, p_lists in p_matrix.items():
                confusion_matrix(s, 'bundle_{}_p-value'.format(electrode_bundle), p_lists, [k for k in condition_dict.keys()], plot_time_points, condition, plot_path)
        ### Preparing data for the final plot for non-searchlight
        else:
            for condition, electrode_dict in all_p_values.items():
                for electrode_name, electrode_tuples in electrode_dict.items():
                    final_plot[condition].append([k[1] for k in electrode_tuples])

    ### Plotting the significant points for all conditions together, for non-searchlight data
    if not args.searchlight:
        #basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], t_values, 't', plot_path)
        basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], all_p_values, 'p', plot_path, counts)
    else:
        pass
        

### Plotting the across subject average for the non-searchlight condition
if not args.searchlight:
    final_plot = {k : numpy.nanmean(v, axis=0) for k, v in final_plot.items()}
    plot_path = os.path.join('p-value_plots', 'rsa_searchlight_{}_{}_{}_{}'.format(True if args.searchlight else False, args.analysis, args.word_selection, args.computational_model))
    basic_scatter_plot_all_electrodes_subject_p_values('all_subjects', [timepoint_converter[t] for t in time_points], final_plot, 'average_p', plot_path)
