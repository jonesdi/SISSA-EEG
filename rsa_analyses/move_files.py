import argparse
import os

from rsa_utils import prepare_folder

parser = argparse.ArgumentParser()
parser.add_argument('--searchlight', action='store_true', default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='objective_accuracy', choices=['objective_accuracy', 'subjective_judgments'], help='Indicates which pairwise similarities to compare, whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', choices=['all_words', 'targets_only'], help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--computational_model', default='w2v', choices=['w2v', 'original_cooc'], help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

for s in range(3, 17):

    for permutation in range(1, 301):
        out_fold = prepare_folder(args, s, permutation=permutation)
        old_fold = 'rsa_maps/rsa_searchlight_True_objective_accuracy_targets_only_w2v_permutation_True/sub-{:02}_{:03}'.format(s, permutation)
        
        os.makedirs(out_fold, exist_ok=True)
        for f in os.listdir(old_fold):
            old_file = os.path.join(old_fold, f)
            os.system('mv {} {}'.format(old_file, out_fold))
    
    import pdb; pdb.set_trace()
