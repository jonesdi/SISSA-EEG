import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse
import random
from multiprocessing import Process

from tqdm import tqdm

from io_utils import ComputationalModels
from rsa_utils import rsa_per_subject

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

    #rsa_per_subject(args, 3, computational_model)
    processes = list()
    for s in range(3, 17): 
        proc = Process(target=rsa_per_subject, args=(args, s, computational_model,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
