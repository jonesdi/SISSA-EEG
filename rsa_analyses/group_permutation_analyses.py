import argparse
import os
import collections
import numpy
import re
from scipy import stats

from searchlight import SearchlightClusters
from io_utils import EvokedResponses
from plot_utils import plot_two, plot_three, confusion_matrix
from rsa_utils import prepare_folder

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

electrode_index_to_code = SearchlightClusters().index_to_code
timepoint_converter = EvokedResponses(3).time_points
if args.searchlight:
    searchlight_converter = {i : v for i, v in enumerate([t for t in range(0, len(timepoint_converter), 2)])}

final_plot = collections.defaultdict(list)
true_data = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(list)))
permuted_data = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(list)))

permutations = [k for k in itertools.product([-1,1], repeat=14) if -1 in k]

for s in tqdm(range(3, 17)):
    ### Collecting the true results
    sub_dict = dict()

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
                        time_point = timepoint_converter[searchlight_converter[t]
                        true_data[condition]['true'][time_point][elec_code].append(true_value)

                    for perm_index, perm in enumerate(permutations):
                        relevant_index = s - 3
                        if perm[relevant_index] < 1:
                            perm_values = [k*-1. for k in true_values]
                            for t, perm_value in enumerate(perm_values):
                                time_point = timepoint_converter[searchlight_converter[t]
                                permuted_data[condition][perm_index+1][time_point][elec_code].append(perm_value)

        except FileNotFoundError:
            pass

plot_time_point = timepoint_converter[searchlight_converter[time_point]]
plot_path = re.sub('true$', '', prepare_folder(args, s).replace('rsa_maps', 'group_permutation_results'))
os.makedirs(plot_path, exist_ok=True)

thresholds = collections.defaultdict(lambda : collections.defaultdict(list))

for condition, condition_dict in permuted_data.items():
    for perm_index, perm_dict in condition_dict.items():
        for time_point, time_dict in perm_dict.items():
            perm_list = list()
            for elec_code, elec_list in time_point_dict.items():
                perm_list.append(scipy.stats.ttest_1samp(eleclist, popmean=0.0)[0])
            perm_max = max(perm_list)
            thresholds[condition][time_point].append(perm_max)

results = collections.defaultdict(lambda : collections.defaultdict(lambda: collections.defaultdict(list)))

for condition, condition_dict in true_data.items():
    for perm_index, perm_dict in condition_dict.items():
        for time_point, time_dict in perm_dict.items():
            for elec_code, elec_list in time_point_dict.items():
                elec_score = scipy.stats.ttest_1samp(eleclist, popmean=0.0)[0]
                maximal_distribution = thresholds[condition][time_point]
                p_value = 1.-(stats.percentileofscore(maximal_distribution, elec_score)/100.)
                results[condition][elec_code][time_point] = p_value

import pdb; pdb.set_trace()

#t_values = collections.defaultdict(lambda: collections.defaultdict(list))
all_p_values = collections.defaultdict(lambda: collections.defaultdict(list))
rho_matrix = collections.defaultdict(lambda: collections.defaultdict(list))
p_matrix = collections.defaultdict(lambda: collections.defaultdict(list))

for electrode_name, electrode_dict in results.items(): 

    ### We want to have two files:
    # 1. Write to file the significant points (one condition per file)
    # 2. Plot the conditions against one another marking the significant points

    ### We want to prepare for the third file:
    # 3. Collect the per-electrode p-values

    if args.searchlight:
        subject_electrodes_plot = collections.defaultdict(list)

    current_electrode_ps = dict()
    current_electrode_rhos = dict()
    current_permutation_rhos = dict()

    for condition, condition_dict in electrode_dict.items():

        if counts[condition] >= 2:

            original_rhos = list()
            permutation_rhos = list()
            p_values = list()

            ### File 1
            with open(os.path.join(plot_path, '{}_{}_significant_time_points.txt'.format(electrode_name, condition)), 'w') as o:
                o.write('Condition\ttime point\tp-value\tt-value\n')

            time_points = [t for t in range(len(condition_dict['true']))]
            if args.searchlight:
                plot_time_points = [timepoint_converter[searchlight_converter[t]] for t in time_points]
            else:
                plot_time_points = [timepoint_converter[t] for t in time_points]

            for t in time_points:

                if args.searchlight:
                    readable_time_point = timepoint_converter[searchlight_converter[t]]
                else:
                    readable_time_point = timepoint_converter[t]
                rho_at_t = float(condition_dict['true'][t])
                rho_scores = [float(v[t]) for k, v in condition_dict.items() if k!='true']
                p_value = 1.-(stats.percentileofscore(rho_scores, rho_at_t)/100.)
                t_value = stats.t.ppf(1.-(p_value), 299)

                if p_value <= .05:
                    #t_values[condition]['all_electrodes'].append((readable_time_point, t_value))

                    ### File 1
                    with open(os.path.join(plot_path, '{}_{}_significant_time_points.txt'.format(electrode_name, condition)), 'a') as o:
                        o.write('{}\t{}\t{}\t{}\n'.format(condition, readable_time_point, p_value, t_value))
                    #if args.searchlight:
                        #subject_electrodes_plot[electrode_name].append(readable_time_point)

                original_rhos.append(rho_at_t)
                permutation_rhos.append(rho_scores)
                p_values.append((readable_time_point, p_value))


            ### Collecting p-values for file 2
            current_electrode_ps[condition] = p_values
            current_electrode_rhos[condition] = original_rhos
            current_permutation_rhos[condition] = permutation_rhos

            ### Collecting p-values for file 3
            p_values_only = [k[1] for k in p_values]
            final_plot[condition].append(p_values_only)
            rho_matrix[condition][electrode_name] = original_rhos
            p_matrix[condition][electrode_name] = p_values_only

    ### File 2: plotting conditions against one another
    plot_two(s, electrode_name, args.computational_model, plot_path, current_electrode_ps, current_electrode_rhos, current_permutation_rhos, time_points, counts)

    #line_and_scatter_plot_all_electrodes_subject_p_values(s, plot_time_points, p_values, original_rhos, permutation_rhos, condition, plot_path, electrode_name, counts)

    ### Collecting p-values for file 3
    #all_p_values[electrode_name] = current_electrode_ps
    #final_plot[condition][electrode_name] = p_values
    all_p_values[condition][electrode_name] = current_electrode_ps

### Plotting file 3 (searchlight), the confusion matrix for the searchlight cluster results
if args.searchlight:
    #subject_electrodes_scatter_plot(s, plot_time_points, subject_electrodes_plot, condition, plot_path)
    for condition, condition_dict in rho_matrix.items():
        for electrode in ['A', 'B', 'C', 'D']:
            electrode_names = ['{}{}'.format(electrode, i) for i in range(1, 33)]
            rho_electrodes = [condition_dict[e] for e in electrode_names]
            confusion_matrix(s, electrode, 'rho', args.computational_model, rho_electrodes, condition, plot_path)

    for condition, condition_dict in p_matrix.items():
        for electrode in ['A', 'B', 'C', 'D']:
            electrode_names = ['{}{}'.format(electrode, i) for i in range(1, 33)]
            p_electrodes = [condition_dict[e] for e in electrode_names]
            confusion_matrix(s, electrode, 'p', args.computational_model, p_electrodes, condition, plot_path)

'''
### Preparing data file 3 (non-searchlight), the final plot for non-searchlight
else:
    final_plot = collections.defaultdict(list)
    for condition, electrode_dict in all_p_values.items():
    
        for electrode_name, electrode_tuples in electrode_dict.items():
            final_plot[condition].append([k[1] for k in electrode_tuples])
'''
'''
### Plotting the significant points for all conditions together, for non-searchlight data
if not args.searchlight:
    #basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], t_values, 't', plot_path)
    basic_scatter_plot_all_electrodes_subject_p_values(s, [timepoint_converter[t] for t in time_points], all_p_values, 'p', plot_path, counts)
else:
    pass
'''
### File 3 (non-searchlight)
# Plotting the across subject average for the non-searchlight condition
if not args.searchlight:
    final_plot = {k : numpy.nanmean(v, axis=0) for k, v in final_plot.items()}
    final_plot_path = re.sub('sub-.+', '', prepare_folder(args, s).replace('rsa_maps', 'permutation_results'))
    plot_three('all_subjects', [timepoint_converter[t] for t in time_points], final_plot, 'average_p', final_plot_path)
