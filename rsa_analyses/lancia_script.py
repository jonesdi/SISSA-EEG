import os
import sys

models = ['ppmi', 'w2v', 'new_cooc', 'original_cooc', 'wordnet', 'cslb', 'orthography', 'visual', 'CORnet']
analyses = ['objective_accuracy', 'subjective_judgments']
#words = ['all_words', 'targets_only']
words = ['all_words']
for analysis in analyses:
    for word in words:
        for m in models:
            print('python3 group_level_analyses.py --minimum_ERPs 2 --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
            os.system('python3 group_level_analyses.py  --minimum_ERPs 2 --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
