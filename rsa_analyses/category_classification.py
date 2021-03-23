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

from tqdm import tqdm
from matplotlib import pyplot

from io_utils import EvokedResponses, ComputationalModels
from searchlight import SearchlightClusters, run_searchlight
from rsa_utils import restrict_evoked_responses, prepare_folder

def run_classification(args, s, evoked_responses, all_time_points):

    categories = ComputationalModels().categories
    
    ### Selecting evoked responses for the current pairwise similarity computations
    selected_evoked = restrict_evoked_responses(args, evoked_responses)

    subject_results = collections.defaultdict(dict)
    subject_info = collections.defaultdict(lambda : collections.defaultdict(list))

    results = collections.defaultdict(list)
    for condition, evoked_dict in selected_evoked.items():
        print(condition)
        #print('Current condition: {}'.format(condition))
        all_data = collections.defaultdict(list)
        
        for k, v in evoked_dict.items():
            for vec in v:
                all_data[categories[k]].append(numpy.array(vec))
        ### Checking that there is enough data to carry out the evaluation
        if len(all_data.keys()) == 2 and min([len(v) for k, v in all_data.items()]) > 20:
            max_data = min([len(v) for k,v in all_data.items()])
            shuffled_data = {k : random.sample(v, k=max_data) for k, v in all_data.items()}
            indices = [i for i in range(max_data)]
            test_indices = [indices[i::10] for i in range(10)]

            scores = list()

            for t in tqdm(range(len(all_time_points))):

                t_scores = list()

                for test_list in test_indices:

                    train_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in range(max_data) if i not in test_list])
                    train_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in range(max_data) if i not in test_list])

                    test_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in test_list])
                    test_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in test_list])
              
                    svm_model = sklearn.svm.SVC().fit(train_samples[:, :, t], train_labels)
                    score = svm_model.score(test_samples[:, :, t], test_labels)
                    t_scores.append(score)
                scores.append(numpy.nanmean(t_scores))

            results[condition].append(scores)    

    return results


parser = argparse.ArgumentParser()
parser.add_argument('--analysis', default='both_worlds', choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
args = parser.parse_args()

if __name__ == '__main__':

    folder = 'classification_plots'
    os.makedirs(folder, exist_ok=True)

    full_results = collections.defaultdict(list)
    for s in range(2, 17): 

        evoked_responses = EvokedResponses(s)
        all_time_points = evoked_responses.time_points
        sub_results = run_classification(args, s, evoked_responses, all_time_points)

        ### Adding to the final dictionary
        for k, v in sub_results.items():
            full_results[k].append(v)

    ### Plotting results

    times = [v for k, v in all_time_points.items()]
    for k, v in full_results.items():
        fig, ax = pyplot.subplots()
        y = numpy.nanmean([vec[0] for vec in v], axis=0)
        for vec in v:
            #ax.plot(vec[0])
            ax.scatter(x=times, y=vec[0], alpha=0.2)
        ax.plot(times, y)
        ax.hlines(y=0.5, xmin=times[0], xmax=times[-1], linestyles='dashed', linewidths=.5, colors='darkgray')
        ax.set_title('Accuracies for {}'.format(k))
        pyplot.savefig(os.path.join(folder, '{}.png'.format(k)), dpi=300) 
