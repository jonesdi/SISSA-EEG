import collections
import itertools
import mne
import numpy
import os
import random

class ExperimentInfo:

    def __init__(self, args):
      
        self.n_subjects, self.paths = self.find_subjects(args)
        self.words_to_cats = self.read_stimuli(args)
        self.cats_to_words = {v : k for k, v in self.words_to_cats.items()}
    
    def find_subjects(self, args):
        path = args.data_folder
        n_subjects = 0
        paths = list()
        for root, direc, files in os.walk(path):
            for f in files:
                if 'epo.fif' in f:
                    file_path = os.path.join(root, \
                                             f)
                    assert os.path.exists(file_path)
                    events_path = file_path.replace('fif', 'events')
                    assert os.path.exists(events_path)
                    n_subjects += 1
                    paths.append((file_path, events_path))

        return n_subjects, paths

    def read_stimuli(self, args):
        experiment_path = os.path.join('lab', \
                                    'lab_{}'.format(args.experiment_id))
        if args.experiment_id == 'one':
            file_name = 'stimuli_final.csv'
            separator = ';'
            cat_index = 1
        elif args.experiment_id == 'two':
            file_name = 'chosen_words.txt'
            separator = '\t'
            cat_index = 2

        stimuli_file = os.path.join(experiment_path, file_name)
        with open(stimuli_file) as i:
            lines = [l.strip().split(separator) \
                       for l in i.readlines()][1:]
        if args.experiment_id == 'one':
        
            ### Removing unused words
            ordered_indices = list(range(10)) + list(range(20, 30)) + list(range(10, 20)) + list(range(40, 50))

            lines = [lines[i] for i in ordered_indices]

        words_to_cats = {l[0] : l[cat_index] for l in lines}

        return words_to_cats

class SubjectData:

    def __init__(self, experiment_info, n, args):
        self.subject = n
        self.exp = experiment_info
        self.eeg_path = experiment_info.paths[n][0]
        self.events_path = experiment_info.paths[n][1]
        self.words, self.accuracies, self.reports = self.get_events(args)
        self.eeg_data, self.times, self.permutations = self.get_eeg_data(args)

    def get_events(self, args):
        awareness_mapper = {'1' : 'low', \
                            '2' : 'medium', \
                            '3' : 'high'}
        with open(self.events_path) as i:
            events = [l.strip().split('\t') for l in i.readlines()][1:]
        if args.experiment_id == 'two':
            events = [l for l in events if len(l) == 3] 
        words = [l[0] for l in events]
        accuracies = [l[1] for l in events]
        reports = [awareness_mapper[l[2]] for l in events]

        return words, accuracies, reports

    def get_eeg_data(self, args):

        epochs = mne.read_epochs(self.eeg_path, verbose=False)
        times = epochs.times

        #assert len(epochs) == len(self.words)

        if args.data_split == 'objective_accuracy':
            splits = list(set(self.accuracies))
            indices = {s : [i for i, v in enumerate(self.accuracies) if v==s] for s in splits}

        elif args.data_split == 'subjective_judgments':
            splits = list(set(self.reports))
            indices = {s : [i for i, v in enumerate(self.reports) if v==s] for s in splits}

        #assert numpy.sum([len(v) for k, v in indices.items()]) == len(self.words)

        eeg_data = collections.defaultdict(lambda : collections.defaultdict(list))
        for s, inds in indices.items():
            for ind in inds:
                word = self.words[ind]
                epoch = epochs[ind]
                eeg_data[s][word].append(epoch)
                
        assert numpy.sum([len(vec) for k, v in eeg_data.items() for k_two, vec in v.items()]) == \
               len(self.words)

        ### Turning the defaultdict into a regular one
        ### Keeping only words having at least 4 ERPs

        regular_dict = dict()
        for k, v in eeg_data.items():
            word_dict = dict()
            for k_two, v_two in v.items():
               #if k_two in final_words:
               if len(v_two) >= 4:
                   v_two = [vec.get_data()[0,:,:] for vec in v_two]
                   # Shuffling the vectors so as to break temporal corr
                   #v_two = random.sample(v_two, k=len(v_two))
                   word_dict[k_two] = v_two

            regular_dict[k] = word_dict

        ### Averaging 4 ERPs
        '''
        max_dict = dict()
        for w in final_words:
            max_dict[w] = min([len(v[w]) for k, v in regular_dict.items()])
        '''
        '''
        ### If we're not running a searchlight, then
        ### subsample by averaging 5 time points

        if args.analysis == 'classification':
            relevant_indices = [i for i in range(times.shape[0])][::5]
            relevant_indices = [i for i in relevant_indices if i+5<times.shape[0]]
        '''

        final_dict = dict()
        for k, v in regular_dict.items():
            k_dict = dict()
            for w, vecs in v.items():

                '''
                ### Subsampling average happens here
                if args.analysis == 'classification':
                    new_vecs = list()
                    for vec in vecs:
                        vec = numpy.array([numpy.average(\
                                          vec[:, i:i+5], axis=1) \
                                          for i in relevant_indices]).T
                    new_vecs.append(vec)
                else:
                    new_vecs = vecs.copy()
                '''
                new_vecs = vecs.copy()

                # Reducing the number of repetitions?
                #n_repetitions = 4
                #new_vecs = new_vecs[:n_repetitions]

                new_vecs = numpy.average(new_vecs, axis=0)
                k_dict[w] = new_vecs
            final_dict[k] = k_dict

        '''
        ### Correcting times if average subsampling happened
        if args.analysis == 'classification':
            times = times[relevant_indices]
        '''
        if 'classification' not in args.analysis:
            permutations_dict = dict()
        else:
            class_dict = dict()
            permutations_dict = dict()

            cats = list(self.exp.cats_to_words.keys())
       
            for awareness, vecs in final_dict.items():    

                current_data = {c : list() for c in cats}
                for w, vec in vecs.items():
                    current_data[self.exp.words_to_cats[w]].append(vec)
               
                ### Shuffling the exemplars within each category
                current_data = {k : random.sample(v, k=len(v)) for k, v in current_data.items()}

                ### Computing the test combinations

                total_number_words = min([len(v) for k, v in current_data.items()])
                #if total_number_words >= 5: ### Employing conditions with at least 5 words
                if total_number_words < 10: ### Employing conditions with at least 10 words
                    if 'searchlight' not in args.analysis:
                        print('not enough words for: sub {}, condition {}'.format(self.subject+1, awareness))
                else:
                    current_data = {k : v[:total_number_words] for k, v in current_data.items()}

                    number_test_samples = max(1, round(0.2 * total_number_words), 0)
                    
                    #if number_test_samples % 2 == 1:
                    #    number_test_samples += 1
                    #assert number_test_samples % 2 == 0
                    word_splits = list(itertools.combinations(list(range(total_number_words)), number_test_samples))
                    test_splits = list(itertools.product(word_splits, repeat=2))
                    test_splits = random.sample(test_splits, \
                                           k=len(test_splits))

                    del word_splits
                    folds = 8192 if \
                            args.analysis == 'classification' \
                            else 1024
                    test_splits = test_splits[:folds]

                    class_dict[awareness] = current_data
                    permutations_dict[awareness] = test_splits
            
            del final_dict
            final_dict = class_dict

        return final_dict, times, permutations_dict

class ComputationalModel:
 
    def __init__(self, args):
    
        self.model = args.computational_model
        self.word_sims = self.load_word_sims(args)

    def load_word_sims(self, args):

        path = os.path.join('computational_models', 'similarities', \
                            args.experiment_id, '{}.sims'.format(self.model))
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        word_sims = {(sim[0], sim[1]) : float(sim[2]) for sim in lines}

        return word_sims

    def compute_pairwise(self, words):
        
        ordered_words = sorted(words)
        combs = list(itertools.combinations(ordered_words, 2))
        pairwise_similarities = list()
        for c in combs:
            sim = self.word_sims[c]
            pairwise_similarities.append(sim)
        
        return ordered_words, combs, pairwise_similarities
