import os

analyses = ['subjective_judgments', 'objective_accuracy']

for analysis in analyses:
    print('python3 time_resolved_classification.py --analysis {}'.format(analysis))
    os.system('python3 time_resolved_classification.py --analysis {} \
              --data_folder /import/cogsci/andrea/github/SISSA-EEG/eeg_preprocessing/preprocessed_data/'.format(analysis))
