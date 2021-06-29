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

from sklearn import svm
from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm
from io_utils import ExperimentInfo, SubjectData

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', required=True, \
                    choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], \
                    help='Indicates which pairwise similarities to compare, \
                          whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='all_words', \
                    choices=['all_words', 'targets_only'], \
                    help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--searchlight', action='store_true', \
                    default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--PCA', action='store_true', \
                    default=False, help='Indicates whether to reduce dimensionality via PCA or not')
parser.add_argument('--data_folder', type=str, required=True, \
                    help='Folder where to find the preprocessed data')
args = parser.parse_args()

exp = ExperimentInfo(args)
cats = list(exp.cats_to_words.keys())

for n in range(exp.n_subjects):
    eeg = SubjectData(exp, n, args)

    data = eeg.eeg_data
    folds = 8000

    out_path = os.path.join('classification_per_subject', \
                            'sub-{:02}'.format(n+1))
    os.makedirs(out_path, exist_ok=True)

    for awareness, vecs in data.items():    

        current_data = {c : list() for c in cats}
        for w, vec in vecs.items():
            current_data[exp.words_to_cats[w]].append(vec)

        total_number_words = min([len(v) for k, v in current_data.items()])
        if total_number_words >= 5: ### Employing conditions with at least 10 words
            current_data = {k : v[:total_number_words] for k, v in current_data.items()}
            number_test_samples = max(1, int(0.1 * total_number_words))
            word_splits = list(itertools.combinations(list(range(total_number_words)), number_test_samples))
            test_splits = list(itertools.product(word_splits, repeat=2))

            scores_times = list()
            for time_i, time in tqdm(enumerate(list(eeg.times))):
                time_list = list()
                for s in test_splits[:folds]:
                
                    train_samples = list()
                    train_labels = list()
                    test_samples = list()
                    test_labels = list()
                    for cat_vecs_i, cat_vecs in enumerate(current_data.items()):
                        current_s = s[cat_vecs_i]
                        for vec_i, vec in enumerate(cat_vecs[1]):
                            if vec_i not in current_s:
                                train_samples.append(vec[:, time_i])
                                train_labels.append(cat_vecs[0])
                            else:
                                test_samples.append(vec[:, time_i])
                                test_labels.append(cat_vecs[0])

                    svm_model = svm.SVC().fit(train_samples, train_labels)
                    accuracy = svm_model.score(test_samples, test_labels)
                    time_list.append(accuracy)
                scores_times.append(numpy.average(time_list))

            with open(os.path.join(out_path, 'sub_{:02}_{}_scores.txt'.format(n+1, awareness)), 'w') as o:
                for t in eeg.times:
                    o.write('{}\t'.format(t))
                o.write('\n')
                for d in scores_times:
                    o.write('{}\t'.format(d))
        else:
            print('not enough words for: sub {}, condition {}'.format(n+1, awareness))
'''
numpy.seterr(all='raise')

if __name__ == '__main__':

    folder = os.path.join('time_resolved_classification_plots', args.analysis)
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

        number_of_subjects = len(v)

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
        ax.legend(['subjects N={}'.format(number_of_subjects)])
        ax.hlines(y=0.5, xmin=all_time_points[0], xmax=times[-1], linestyles='dashed', linewidths=.1, colors='darkgray')
        pca_marker = ' - using PCA' if args.PCA else ''
        ax.set_title('Accuracies for {}{} - {}'.format(k, pca_marker, args.word_selection))
        pca_marker = '_PCA' if args.PCA else ''
        pyplot.savefig(os.path.join(folder, '{}{}_{}.png'.format(k, pca_marker, args.word_selection)), dpi=300) 
        pyplot.clf()
import sys
sys.path.append('../rsa_analyses')

from io_utils import EvokedResponses, ComputationalModels
from searchlight import SearchlightClusters, run_searchlight
from rsa_utils import restrict_evoked_responses, prepare_folder
def run_classification(args, s, evoked_responses, t_points):

    categories = ComputationalModels().categories
    
    # Selecting evoked responses for the current pairwise similarity computations
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
              
                    current_t_train = train_samples[:, :, t]
                    current_t_test = test_samples[:, :, t]

                    if args.PCA:
                        pca = sklearn.decomposition.PCA(n_components=.99, svd_solver='full').fit(current_t_train)
                        current_t_train = pca.transform(current_t_train)
                        current_t_test = pca.transform(current_t_test)

                    svm_model = svm.SVC().fit(current_t_train, train_labels)
                    score = svm_model.score(current_t_test, test_labels)
                    iteration_scores.append(score)

                all_iterations.append(iteration_scores)

            scores = numpy.average(numpy.array(all_iterations), axis=0)
            #print(scores.shape)
            assert scores.shape[0] == len(t_points)

            results[condition].append(scores)    

    return results
'''
