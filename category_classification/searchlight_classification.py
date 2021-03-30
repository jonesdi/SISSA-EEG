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
import multiprocessing
import logging

from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot
from scipy import stats

import sys
sys.path.append('../rsa_analyses')

from io_utils import EvokedResponses, ComputationalModels
from searchlight import SearchlightClusters, run_searchlight
from rsa_utils import restrict_evoked_responses, prepare_folder

def run_searchlight_classification(arguments):
    args, electrode, electrode_data, t_points, train_indices, test_indices, max_data = arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]

    all_iterations = list()
    ### Shuffling and testing 300 times
    for i in range(300):

        shuffled_data = {k : random.sample(v, k=max_data) for k, v in electrode_data.items()}

        train_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in train_indices])
        train_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in train_indices])

        test_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in test_indices])
        test_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in test_indices])

        iteration_scores = list()

        #for t in tqdm(range(len(all_time_points))):
        for t in t_points:
      
            current_t_train = train_samples[:, :, t:t+args.temporal_window_size].reshape(train_samples.shape[0], -1)
            current_t_test = test_samples[:, :, t:t+args.temporal_window_size].reshape(test_samples.shape[0], -1)

            if args.PCA:
                pca = sklearn.decomposition.PCA(n_components=.99, svd_solver='full').fit(current_t_train)
                current_t_train = pca.transform(current_t_train)
                current_t_test = pca.transform(current_t_test)

            svm_model = sklearn.svm.SVC().fit(current_t_train, train_labels)
            score = svm_model.score(current_t_test, test_labels)
            iteration_scores.append(score)

        all_iterations.append(iteration_scores)

    scores = numpy.average(numpy.array(all_iterations), axis=0)
    #print(scores.shape)
    assert scores.shape[0] == len(t_points)

    return (electrode, scores)

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', default='both_worlds', choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], \
                    help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], \
                    help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--PCA', action='store_true', default=False, \
                    help='Indicates whether to reduce dimensionality via PCA or not')
parser.add_argument('--hop', default=3, type=int, \
                    help='Indicates which similarities to use for comparison to the eeg similarities')
parser.add_argument('--temporal_window_size', default=7, type=int, \
                    help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
### Loading the categories
categories = ComputationalModels().categories

### Loading the searchlight clusters
searchlight_clusters = SearchlightClusters()

if __name__ == '__main__':

    for s in range(2, 17): 


        map_folder = os.path.join('classification_maps', args.analysis, \
                                  'window_{}_hop_{}'.format(args.temporal_window_size, args.hop), \
                                  args.word_selection, 'sub-{:02}'.format(s))
        os.makedirs(map_folder, exist_ok=True)

        evoked_responses = EvokedResponses(s)
        all_time_points = evoked_responses.time_points
        t_points = [t for t in range(0, len(all_time_points), args.hop) if t+args.temporal_window_size<len(all_time_points)]
        
        ### Selecting evoked responses for the current pairwise similarity computations
        selected_evoked = restrict_evoked_responses(args, evoked_responses)

        for condition, evoked_dict in selected_evoked.items():
            logging('Now starting with subject {}, condition {}...'.format(s, condition))           
            #print('Current condition: {}'.format(condition))
            all_data = collections.defaultdict(list)
            
            for k, v in evoked_dict.items():
                for vec in v:
                    all_data[categories[k]].append(numpy.array(vec))
            ### Checking that there is enough data to carry out the evaluation
            if len(all_data.keys()) == 2 and min([len(v) for k, v in all_data.items()]) > 20:

                max_data = min([len(v) for k,v in all_data.items()])
                percentage = int(max_data*0.8)

                train_indices = [i for i in range(max_data) if i<=percentage]
                test_indices = [i for i in range(max_data) if i>percentage]

                arguments = list()
                for center in range(128):

                    relevant_electrode_indices = searchlight_clusters.neighbors[center]
                    relevant_electrode_masks = numpy.array([True if i in relevant_electrode_indices else False for i in range(128)])
                    electrode_data = collections.defaultdict(list)
                    for k, v in all_data.items():
                        for vec in v:
                            electrode_data[k].append(vec[relevant_electrode_masks])
                    arguments.append([args, center, electrode_data, t_points, train_indices, test_indices, max_data])
                    
                with Pool(processes=os.cpu_count()-1) as pool:
                    res = pool.map(run_searchlight_classification, arguments)
                    pool.close()
                    pool.join()
                res = sorted(res, key=lambda item: item[0])
                logging('Now finished with subject {}..., condition {}'.format(s, condition))           

                ### Writing the classification maps
                with open(os.path.join(map_folder, '{}.map'.format(condition)), 'w') as o:
                    o.write('Searchlight cluster index\tAccuracy per time window\n')
                    for elec_res in res:
                        elec = elec_res[0]
                        accs = elec_res[1]
                        o.write('{}\t'.format(elec))
                        for acc in accs:
                            o.write('{}\t'.format(acc))
                        o.write('\n')
