import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

models = ['log_cooc', 'ppmi', 'w2v', 'bert', 'wordnet']
models = ['bert']
data_split = ['objective_accuracy', 'subjective_judgments']

for d in data_split:
    for m in models:
        logging.info([d, m])
        os.system('python3 main.py '\
                    #'--analysis group_searchlight '\
                    #'--analysis behavioural '\
                    #'--analysis classification '\
                    #'--analysis classification_searchlight '\
                    '--analysis group_classification_searchlight '\
                    #'--analysis group_rsa_searchlight '\
                    #'--analysis rsa_searchlight '\
                    '--data_split {} '\
                    #'--computational_model {} '\
                    '--data_folder /import/cogsci/andrea/dataset/neuroscience/conscious_unconscious_processing/two/preprocessed_data '\
                    #'--data_folder /import/cogsci/andrea/dataset/neuroscience/conscious_unconscious_processing/two/raw_data '\
                    #'--plot '\
                    '--experiment_id two'.format(d, m))
