import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

models = ['cooc', 'log_cooc', 'ppmi', 'w2v', 'bert', 'wordnet']
models = ['cooc']
data_split = ['objective_accuracy', 'subjective_judgments']

for d in data_split:
    for m in models:
        logging.info([d, m])
        os.system('python3 main.py '\
                    '--analysis rsa_searchlight '\
                    #'--analysis group_searchlight '\
                    '--data_split {} '\
                    '--computational_model {} '\
                    '--data_folder /import/cogsci/andrea/dataset/neuroscience/conscious_unconscious_processing/ '\
                    '--experiment_id one'.format(d, m))
