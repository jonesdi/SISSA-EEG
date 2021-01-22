import collections
import mne
import os

### Importing computational models

class ComputationalModels:

    def __init__(self):

        self.words = self.read_stimuli()
        self.w2v = self.get_w2v_sims()
        self.original_cooc = self.get_original_cooc()
        assert {k : '' for k in self.w2v.keys()} == {k : '' for k in self.original_cooc.keys()}

    def read_stimuli(self):
        stimuli = list()
        with open('../lab_experiment/stimuli_final.csv', 'r') as stimuli_file:
            for i, l in enumerate(stimuli_file):
                if i > 0: 
                    l = l.strip().split(';')
                    stimuli.append(l[0])
        return stimuli

    def get_w2v_sims(self):

        w2v_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'w2v', 'w2v_max_vocab_150000.sims'), 'r') as w2v_file:
            for l in (w2v_file):
                l = l.strip().split('\t')
                if l[0] in self.words and l[1] in self.words:
                    w2v_similarities[l[0]][l[1]] = float(l[3])
                    w2v_similarities[l[1]][l[0]] = float(l[3])

        # Turning defaultdict into a regular dict
        w2v_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in w2v_similarities.items()}

        return w2v_similarities

    def get_original_cooc(self):

        cooc_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'cooc', 'cooc_original.csv'), 'r') as cooc_original_file:
            for i, l in enumerate(cooc_original_file):
                if i > 0: 
                    l = l.strip().split(';')
                    if l[0] in self.words and l[1] in self.words:
                        cooc_original_similarities[l[0]][l[1]] = float(l[2])
                        cooc_original_similarities[l[1]][l[0]] = float(l[2])

        # Turning defaultdict into a regular dict
        cooc_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in cooc_original_similarities.items()}

        return cooc_original_similarities

    def get_new_cooc(self):

        cooc_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'cooc', 'cooc_original.csv'), 'r') as cooc_original_file:
            for i, l in enumerate(cooc_original_file):
                if i > 0: 
                    l = l.strip().split(';')
                    if l[0] in self.words and l[1] in self.words:
                        cooc_original_similarities[l[0]][l[1]] = float(l[2])
                        cooc_original_similarities[l[1]][l[0]] = float(l[2])

        # Turning defaultdict into a regular dict
        cooc_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in cooc_original_similarities.items()}

        return cooc_original_similarities

### Collects and organizes the evoked responses

class EvokedResponses:

    def __init__(self, subject_number):

        self.folder = '/import/cogsci/andrea/dataset/neuroscience/conscious_unconscious_processing/preprocessed_files/sub-{:02}'.format(subject_number)
        self.subject_number = subject_number
        self.events = self.read_events()
        self.original_epochs = self.read_original_epochs()
        self.time_points = self.read_time_points()

    def read_events(self):

        events_dict = collections.defaultdict(list)
        certainty_mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}

        with open(os.path.join(self.folder, 'sub-{:02}_events_rejected_or_good.txt'.format(self.subject_number))) as f:
            event_index = 0
            for l in f:
                l = l.strip().split('\t')
                if l[5] != 'rejected':
                    events_dict[event_index] = [l[0], l[1], l[3], certainty_mapper[int(l[4])]]
                    event_index += 1

        return events_dict

    def read_original_epochs(self):

        epochs = mne.read_epochs(os.path.join(self.folder, 'sub-{:02}_highpass-100Hz-epoched-concatenated.fif'.format(self.subject_number)))
        epochs = epochs.get_data()
        feature_standardizer = mne.decoding.Scaler(scalings='mean')
        epochs = feature_standardizer.fit_transform(epochs)

        return epochs

    def read_time_points(self):

        sampling_points = self.original_epochs.shape[-1]
        time_step = (sampling_points/204.8)/sampling_points
        time_points = collections.defaultdict(float)

        for i in range(sampling_points):
            if i == 0:
                time_points[i] = -0.2
            else:
                current_time = time_points[i-1] + time_step
                if current_time <= 1.1:
                    time_points[i] = current_time
                else:
                    break
        return time_points

