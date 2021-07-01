import argparse

from io_utils import ExperimentInfo, SubjectData
from classification.time_resolved_classification import run_classification

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')

parser.add_argument('--analysis', required=True, \
                    choices=['classification', \
                             'rsa_searchlight'], \
                    help='Indicates which analysis to perform')

parser.add_argument('--data_split', required=True, \
                    choices=['objective_accuracy', \
                             'subjective_judgments', \
                             'both_worlds'], \
                    help='Indicates which pairwise similarities \
                          to compare, whether by considering \
                          objective accuracy or subjective judgments')

parser.add_argument('--searchlight', action='store_true', \
                    default=False, help='Indicates whether to run \
                                         a searchlight analysis or not')

parser.add_argument('--data_folder', type=str, required=True, \
                    help='Folder where to find the preprocessed data')

### Obsolete arguments
'''
parser.add_argument('--word_selection', default='all_words', \
                    choices=['all_words', 'targets_only'], \
                    help='Indicates whether to use \
                          for the analyses only the targets \
                          or all the words')

parser.add_argument('--PCA', action='store_true', \
                    default=False, help='Indicates whether to reduce \
                                         dimensionality via PCA or not')
'''

args = parser.parse_args()

exp = ExperimentInfo(args)
for n in range(exp.n_subjects):
    eeg = SubjectData(exp, n, args)
    import pdb; pdb.set_trace()
    if args.analysis == 'classification':
        run_classification(exp, eeg, n, args)

