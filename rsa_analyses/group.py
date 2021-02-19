import argparse
import os
import collections
import numpy
import re
import logging
import itertools
import pickle
import scipy
import multiprocessing
from scipy import stats

from searchlight import SearchlightClusters
from io_utils import EvokedResponses
from plot_utils import plot_two, plot_three, confusion_matrix
from rsa_utils import prepare_folder

from tqdm import tqdm

def run_sign_perm(input_tuple):

    true_data = input_tuple[0]
    permutation = input_tuple[1]

    thresholds = dict()

    for condition, condition_dict in true_data.items():
        #cond = dict()
        cond = list()
        for time_point, time_dict in condition_dict.items():
            perm_list = list()
            for elec_code, elec_list in time_dict.items():
                perm_elec = [t*p for t, p in zip(elec_list, permutation)]
                perm_list.append(scipy.stats.ttest_1samp(perm_elec, popmean=0.0, nan_policy='omit')[0])
            perm_max = max(perm_list)
            cond.append(perm_max)
        thresholds[condition] = max(cond)

    return thresholds

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--searchlight', action='store_true', default=True, help='Indicates whether to run a searchlight analysis or not')
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
true_data = collections.defaultdict(lambda: collections.defaultdict(lambda : collections.defaultdict(list)))


logging.info('Now loading the data and permuting it...')

for s in tqdm(range(3, 17)):
    ### Collecting the true results

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

plot_path = re.sub('true$', '', prepare_folder(args, s).replace('rsa_maps', 'group_permutation_results'))
os.makedirs(plot_path, exist_ok=True)

### Turning lambdaed dicts into normal dicts
logging.info('Now turning the lambda dictionary into a basic one...')

true_dict = dict()
for condition, condition_dict in true_data.items():
    cond = dict()
    for time_point, time_dict in condition_dict.items():
        time = dict()
        for elec_code, elec_list in time_dict.items():
            time[elec_code] = elec_list
        cond[time_point] = time
    true_dict[condition] = cond

permutations = [k for k in itertools.product([-1,1], repeat=14) if -1 in k]
arguments = [[true_dict, permutation] for permutation in permutations]

logging.info('Now running permutations...')
if __name__ == '__main__':

    pool = multiprocessing.Pool()
    results_collector = pool.imap(func=run_sign_perm, iterable=arguments)
    pool.close()
    pool.join()

logging.info('Now joining the permutations...')
perm_dict = collections.defaultdict(list)
for res in results_collector:
    for condition, perm_max in res.items():
        perm_dict[condition].extend([perm_max])

logging.info('Now running the analyses...')
results = collections.defaultdict(lambda : collections.defaultdict(list))
for condition, condition_dict in true_data.items():
    maximal_distribution = perm_dict[condition]
    for time_point, time_dict in condition_dict.items():
        for elec_code, elec_list in time_dict.items():
            elec_score = scipy.stats.ttest_1samp(elec_list, popmean=0.0, nan_policy='omit')[0]
            p_value = 1.-(stats.percentileofscore(maximal_distribution, elec_score)/100.)
            if p_value/2 <= .05:
                print([elec_code, time_point])
            results[condition][elec_code][time_point] = p_value
