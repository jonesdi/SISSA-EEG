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
from scipy import stats

from io_utils import EvokedResponses, ComputationalModels
from searchlight import SearchlightClusters, run_searchlight
from rsa_utils import restrict_evoked_responses, prepare_folder

def run_classification(args, s, evoked_responses, t_points):

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
            percentage = int(max_data*0.8)

            train_indices = [i for i in range(max_data) if i<=percentage]
            #test_indices = [indices[i::10] for i in range(10)]
            test_indices = [i for i in range(max_data) if i>percentage]

            ### Averaging 7 time points so as to reduce computation time
            compressed_data = dict()

            for k, v in all_data.items():
                vec_list = list()
                for vec in v:
                    new_vec = numpy.array([numpy.average(vec[:, i:i+7], axis=1) for i in t_points])
                    new_vec = numpy.transpose(new_vec)
                    assert new_vec.shape == (vec.shape[0], len(t_points))
                    vec_list.append(new_vec)
                compressed_data[k] = vec_list
            
            all_iterations = list()
            ### Shuffling and testing 300 times
            for i in tqdm(range(300)):

                shuffled_data = {k : random.sample(v, k=max_data) for k, v in compressed_data.items()}

                train_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in train_indices])
                train_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in train_indices])

                test_samples = numpy.array([v_list[i] for cat, v_list in shuffled_data.items() for i in test_indices])
                test_labels = numpy.array([cat for cat, v_list in shuffled_data.items() for i in test_indices])

                iteration_scores = list()

                #for t in tqdm(range(len(all_time_points))):
                for t in range(len(t_points)):
              
                    svm_model = sklearn.svm.SVC().fit(train_samples[:, :, t], train_labels)
                    score = svm_model.score(test_samples[:, :, t], test_labels)
                    iteration_scores.append(score)

                all_iterations.append(iteration_scores)

            scores = numpy.average(numpy.array(all_iterations), axis=0)
            print(scores.shape)
            assert scores.shape[0] == len(t_points)

            results[condition].append(scores)    

    return results

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', default='both_worlds', choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--searchlight', action='store_true', default=False, help='Indicates whether to run a searchlight analysis or not')
args = parser.parse_args()

numpy.seterr(all='raise')

if __name__ == '__main__':

    folder = 'classification_plots'
    os.makedirs(folder, exist_ok=True)

    full_results = collections.defaultdict(list)
    #times = [v for k, v in all_time_points.items()]

    for s in range(2, 17): 

        evoked_responses = EvokedResponses(s)
        all_time_points = evoked_responses.time_points
        t_points = [t for t in range(0, len(all_time_points), 7) if t+3 in all_time_points.keys()]
        sub_results = run_classification(args, s, evoked_responses, t_points)

        ### Adding to the final dictionary
        for k, v in sub_results.items():
            full_results[k].append(v)

    times = [all_time_points[i+3] for i in t_points]

    ### Computing p-values
    p_values = dict()
    for k, v in full_results.items():
        k_list = list()
        for t in range(len(times)):
            p_value = stats.ttest_1samp([vec[0][t] for vec in v], popmean=.5, alternative='greater')[1]
            k_list.append(p_value)
        corrected_ps = [i for i, v in enumerate(mne.stats.fdr_correction(k_list)[0]) if v==True]
        
        p_values[k] = corrected_ps

    ### Plotting results

    for k, v in full_results.items():
        fig, ax = pyplot.subplots()
        y = numpy.nanmean([vec[0] for vec in v], axis=0)

        significant_ps = p_values[k]
        significant_xs = list()
        significant_ys = list()
        for i in significant_ps:
            significant_xs.append(times[i])
            significant_ys.append(y[i])
        if len(significant_xs) >= 1:
            ax.scatter(x=significant_xs, y=significant_ys, edgecolors='black', color='white', linewidths=1.)

        for vec in v:
            #ax.plot(vec[0])
            ax.scatter(x=times, y=vec[0], alpha=0.1)
        ax.plot(times, y)
        ax.hlines(y=0.5, xmin=all_time_points[0], xmax=times[-1], linestyles='dashed', linewidths=.1, colors='darkgray')
        ax.set_title('Accuracies for {}'.format(k))
        pyplot.savefig(os.path.join(folder, '{}.png'.format(k)), dpi=300) 
        pyplot.clf()


    import pdb; pdb.set_trace()
