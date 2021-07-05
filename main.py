import argparse
import multiprocessing
import os
import scipy

from tqdm import tqdm

from io_utils import ComputationalModel, ExperimentInfo, SubjectData
from classification.time_resolved_classification import run_classification
from rsa.group_searchlight import run_group_searchlight
from rsa.rsa_searchlight import finalize_rsa_searchlight, run_searchlight
from searchlight.searchlight_utils import SearchlightClusters

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')

parser.add_argument('--analysis', required=True, \
                    choices=['classification', \
                             'rsa_searchlight', \
                             'group_searchlight'], \
                    help='Indicates which analysis to perform')

parser.add_argument('--computational_model', required=False, \
                    choices=['cooc', 'log_cooc', 'ppmi', \
                             'w2v', 'bert', 'wordnet'], \
                    help='Which model?')

parser.add_argument('--data_split', required=True, \
                    choices=['objective_accuracy', \
                             'subjective_judgments', \
                             'both_worlds'], \
                    help='Indicates which pairwise similarities \
                          to compare, whether by considering \
                          objective accuracy or subjective judgments')

parser.add_argument('--data_folder', type=str, required=True, \
                    help='Folder where to find the preprocessed data')

args = parser.parse_args()

general_output_folder = os.path.join('results', args.analysis, \
                                     args.experiment_id, args.data_split)
os.makedirs(general_output_folder, exist_ok=True)

exp = ExperimentInfo(args)

if args.analysis == 'group_searchlight':
    clusters = SearchlightClusters()
    run_group_searchlight(args, exp, clusters, general_output_folder)

else:

    if args.analysis == 'rsa_searchlight':

        general_output_folder = os.path.join(general_output_folder, args.computational_model)
        os.makedirs(general_output_folder, exist_ok=True)
        if not args.computational_model:
            raise RuntimeError('You need to specify a computational model!')
        comp_model = ComputationalModel(args)
        searchlight_clusters = SearchlightClusters()
        electrode_indices = [searchlight_clusters.neighbors[center] for center in range(128)]

    for n in tqdm(range(exp.n_subjects)):

        eeg = SubjectData(exp, n, args)

        if args.analysis == 'classification':

            run_classification(exp, eeg, n, args)

        if args.analysis == 'rsa_searchlight':

            data = eeg.eeg_data

            times = eeg.times
            relevant_times = [t_i for t_i, t in enumerate(times) if t_i+16<len(times)][::8]
            explicit_times = [times[t] for t in relevant_times]
            clusters = [(e_s, t_s) for e_s in electrode_indices for t_s in relevant_times]

            for awareness, vecs in data.items():    

                words = list(vecs.keys())
                ### Only employing conditions with at least 5 words
                if len(words) >= 5:

                    ordered_words, combs, pairwise_similarities = comp_model.compute_pairwise(words)

                    with multiprocessing.Pool() as p:

                        results = p.map(run_searchlight, \
                                        [[vecs, comp_model, cluster, combs, pairwise_similarities] \
                                                                          for cluster in clusters])
                        p.terminate()
                        p.join()

                    finalize_rsa_searchlight(results, relevant_times, explicit_times, \
                                             general_output_folder,awareness, n)

