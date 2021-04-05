import os
import sys

#models = ['ppmi', 'w2v', 'new_cooc', 'wordnet', 'cslb', 'orthography', 'visual', 'CORnet']
#models = ['visual', 'CORnet', 'cslb', 'orthography']
models = ['ppmi', 'w2v', 'new_cooc', 'original_cooc']
#analyses = ['objective_accuracy', 'subjective_judgments', 'both_worlds']
analyses = ['objective_accuracy']
#words = ['all_words', 'targets_only']
words = ['targets_only']
for analysis in analyses:
    for word in words:
        for m in models:
            print('python3 group_level_analyses.py --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
            os.system('python3 group_level_analyses.py --analysis {} --word_selection {} --computational_model {}'.format(analysis, word, m))
