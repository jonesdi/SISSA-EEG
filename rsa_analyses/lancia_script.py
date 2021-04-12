import os
import sys

models = ['ppmi', 'w2v', 'new_cooc', 'original_cooc', 'wordnet', 'cslb', 'orthography', 'visual', 'CORnet']
analyses = ['objective_accuracy', 'subjective_judgments', 'both_worlds']
words = ['all_words', 'targets_only']
for analysis in analyses:
    for word in words:
        for m in models:
            print('python3 rsa_eeg.py --searchlight --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
            os.system('python3 rsa_eeg.py --searchlight --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
