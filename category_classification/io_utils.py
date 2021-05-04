import os
import mne
import numpy
import collections
import random

class ExperimentInfo:

    def __init__(self, args):
      
        self.n_subjects, self.paths = self.find_subjects(args)
        self.words_to_cats = self.read_stimuli()
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

    def read_stimuli(self):
        stimuli_file = os.path.join('..', \
                                    'lab_experiment', \
                                    'stimuli_final.csv')
        with open(stimuli_file) as i:
            lines = [l.strip().split(';') for l in i.readlines()][1:]
        words_to_cats = {l[0] : l[1] for l in lines}

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
        if args.analysis == 'objective_accuracy':
            splits = list(set(self.accuracies))
            indices = {s : [i for i, v in enumerate(self.accuracies) if v==s] for s in splits}
        elif args.analysis == 'subjective_judgments':
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

        '''
        ### Removing words present less than 4 times in one of the two conditions
        final_words = list()
        for w in set(self.words):
            marker = 0
            for k, v in eeg_data.items():
                if len(v[w]) >= 4:
                    pass
                else:
                    marker += 1
            if marker == 0:
                final_words.append(w)
        '''

        ### Turning the defaultdict into a regular one
        ### Keeping only words having at least 4 ERPs

        regular_dict = dict()
        for k, v in eeg_data.items():
            word_dict = dict()
            for k_two, v_two in v.items():
               #if k_two in final_words:
               if len(v_two) >= 4:
                   v_two = [vec.get_data()[0,:,:] for vec in v_two]
                   shuffled = random.sample(v_two, k=len(v_two))
                   word_dict[k_two] = shuffled

            regular_dict[k] = word_dict

        ### Subsampling and averaging 4 ERPs
        '''
        max_dict = dict()
        for w in final_words:
            max_dict[w] = min([len(v[w]) for k, v in regular_dict.items()])
        '''
        relevant_indices = [i for i in range(times.shape[0])][::5]
        relevant_indices = [i for i in relevant_indices if i+5<times.shape[0]]
        final_dict = dict()
        for k, v in regular_dict.items():
            k_dict = dict()
            for w, vecs in v.items():
                #new_vecs = numpy.average(vecs[:max_dict[w]], axis=0)
                new_vecs = list()
                for vec in vecs[:4]:
                    subsampled_vec = numpy.array([numpy.average(vec[:, i:i+9], axis=1) for i in relevant_indices]).T
                    new_vecs.append(subsampled_vec)
                new_vecs = numpy.average(new_vecs, axis=0)
                k_dict[w] = new_vecs
            final_dict[k] = k_dict
        times = times[relevant_indices]

        return final_dict, times
