import os

analyses = ['objective_accuracy', 'subjective_judgments', 'both_worlds']
words = ['all_words', 'targets_only']

for analysis in analyses:
    for word in words:
        print('python3 time_resolved_classification.py --analysis {} --word_selection {}'.format(analysis, word))
        os.system('python3 group_searchlight_classification_analysis.py --analysis {} --word_selection {}'.format(analysis, word))
