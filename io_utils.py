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

        words_to_cats = {l[0] : cat_index for l in lines}

        return words_to_cats

class SubjectData:

    def __init__(self, experiment_info, n, args):
        self.eeg_path = experiment_info.paths[n][0]
        self.events_path = experiment_info.paths[n][1]
        self.words, self.accuracies, self.reports = self.get_events()
        self.eeg_data, self.times = self.get_eeg_data(args)

    def get_events(self):
        awareness_mapper = {'1' : 'low', \
                  '2' : 'medium', \
                  '3' : 'high'}
        with open(self.events_path) as i:
            events = [l.strip().split('\t') for l in i.readlines()][1:]
        words = [l[0] for l in events]
        accuracies = [l[1] for l in events]
        reports = [awareness_mapper[l[2]] for l in events]

        return words, accuracies, reports

    def get_eeg_data(self, args):

        epochs = mne.read_epochs(self.eeg_path)
        times = epochs.times

        assert len(epochs) == len(self.words)

        if args.data_split == 'objective_accuracy':
            splits = list(set(self.accuracies))
            indices = {s : [i for i, v in enumerate(self.accuracies) if v==s] for s in splits}

        elif args.data_split == 'subjective_judgments':
            splits = list(set(self.reports))
            indices = {s : [i for i, v in enumerate(self.reports) if v==s] for s in splits}

        assert numpy.sum([len(v) for k, v in indices.items()]) == len(self.words)

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
                   shuffled = random.sample(v_two, k=len(v_two))
                   word_dict[k_two] = shuffled

            regular_dict[k] = word_dict

        ### Averaging 4 ERPs
        '''
        max_dict = dict()
        for w in final_words:
            max_dict[w] = min([len(v[w]) for k, v in regular_dict.items()])
        '''
        ### If we're not running a searchlight, then
        ### subsample by averaging 5 time points

        if args.analysis == 'classification':
            relevant_indices = [i for i in range(times.shape[0])][::5]
            relevant_indices = [i for i in relevant_indices if i+5<times.shape[0]]

        final_dict = dict()
        for k, v in regular_dict.items():
            k_dict = dict()
            for w, vecs in v.items():
                new_vecs = list()
                for vec in vecs[:4]:
                    ### Subsampling average happens here
                    if not args.analysis == 'classification':
                        vec = numpy.array([numpy.average(\
                                          vec[:, i:i+5], axis=1) \
                                          for i in relevant_indices]).T
                    new_vecs.append(vec)
                new_vecs = numpy.average(new_vecs, axis=0)
                k_dict[w] = new_vecs
            final_dict[k] = k_dict

        ### Correcting times if average subsampling happened
        if args.analysis == 'classification':
            times = times[relevant_indices]

        return final_dict, times

class ComputationalModel:
 
    def __init__(self, args):
    
        self.model = args.computational_model
        self.word_sims = self.load_word_sims(args)

    def load_word_sims(self, args):

        path = os.path.join('computational_models', 'similarities', \
                            args.experiment_id, '{}.sims'.format(self_model))
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        word_sims = [(sim[0], sim[1]) : float(sim[2]) for sim in lines]

        return word_sims

    def compute_pairwise(self, words):
        
        ordered_words = sorted(words)
        combs = list(itertools.combinations(ordered_words, 2))
        pairwise_similarities = list()
        for c in combs:
            sim = self.word_sims(c)
            pairwise_similarities.append(sim)
        
        return ordered_words, combs, pairwise_similarities
