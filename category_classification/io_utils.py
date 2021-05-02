import os
import mne

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
        self.eeg_data, self.times = self.get_eeg_data()

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

    def get_eeg_data(self):
        epochs = mne.read_epochs(self.eeg_path)
        assert len(epochs) == len(self.words)
        eeg_data = dict()
        for e_i, e in enumerate(epochs):
            label = (self.words[e_i], \
                          self.accuracies[e_i], \
                          self.reports[e_i])
            if label not in eeg_data.keys():
                eeg_data[label] = e.get_data()
            else:
                eeg_data[label].append(e.get_data())
        return eeg_data, times

    def reorganized_data(self, exp, args):

        ### Re-organizing the data according to the current analysis
        if args.analysis == 'objective_accuracy':
            splits = set(exp.accuracies)
        elif args.analysis == 'subjective_accuracy':
            splits = set(exp.reports)
        for split in splits:
            current_split = dict()
            for k, v in self.eeg_data.items():
                if split in k:
                    word = k[0]
                    if word not in current_split.keys():
                        current_split[word] = [v]
                    else:
                        current_split[word].append(v)
            reorganized_data[split] = current_split

        return reorganized_data
