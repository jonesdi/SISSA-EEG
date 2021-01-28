import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse
import random
from multiprocessing import Process

from io_utils import ComputationalModels, EvokedResponses
from rsa_utils import run_rsa

parser = argparse.ArgumentParser()
parser.add_argument('--permutation', action='store_true', default=False, help='Indicates whether to run a permutation analysis or not')
parser.add_argument('--searchlight', action='store_true', default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_model', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

### RSA

if __name__ == '__main__':
    
    ### Loading the computational model
    if args.computational_model == 'w2v':
        computational_model = ComputationalModels().w2v
    elif args.computational_model == 'original_cooc':
        computational_model = ComputationalModels().original_cooc

    #rsa_per_subject(args, 3, computational_model)
    if args.permutation:

        ### Preparing the batches
        batches = list()
        workers = os.cpu_count() - 1
        separators = [i for i in range(1, 301, workers)]
        for i, v in enumerate(separators):
            if v != separators[-1]:
                batches.append([i for i in range(v, separators[i+1])])
            else:
                batches.append([i for i in range(v, 301)])
        for b in batches:
            assert len(b) <= workers


        for s in range(4, 17): 
        #for s in range(12, 17): 

            evoked_responses = EvokedResponses(s)
            all_time_points = evoked_responses.time_points

            for batch in batches:
                processes = list()
                for permutation in batch:

                    #run_rsa(args, s, evoked_responses, computational_model, all_time_points, permutation)
                    
                    proc = Process(target=run_rsa, args=(args, s, evoked_responses, computational_model, all_time_points, permutation, ))
                    processes.append(proc)
                    proc.start()

                for proc in processes:
                    proc.join()
    
    else:
        processes = list()

        for s in range(3, 17): 
            evoked_responses = EvokedResponses(s)
            all_time_points = evoked_responses.time_points

            proc = Process(target=run_rsa, args=(args, s, evoked_responses, computational_model, all_time_points))
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
