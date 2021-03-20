import collections
import mne
import os
import itertools
from matplotlib import pyplot

from scipy import stats
from sklearn.manifold import TSNE

import sys
sys.path.append('/import/cogsci/andrea/github/')

from grano.plot_utils import confusion_matrix

### Importing computational models

class ComputationalModels:

    def __init__(self):

        self.words = self.read_stimuli()
        self.w2v = self.get_w2v_sims()
        self.original_cooc = self.get_original_cooc()
        self.ppmi = self.get_new_cooc(mode='ppmi')
        self.new_cooc = self.get_new_cooc()
        self.wordnet = self.get_wordnet()
        self.orthography = self.get_orthography()
        self.visual = self.get_visual()
        assert {k : '' for k in self.w2v.keys()} == {k : '' for k in self.original_cooc.keys()}

    def read_stimuli(self):
        stimuli = list()
        with open('../lab_experiment/stimuli_final.csv', 'r') as stimuli_file:
            for i, l in enumerate(stimuli_file):
                if i > 0: 
                    l = l.strip().split(';')
                    stimuli.append(l[0])
        return stimuli

    def get_orthography(self):

        orthography_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'orthography', 'orthography.sims'), 'r') as orthography_original_file:
            for i, l in enumerate(orthography_original_file):
                l = l.strip().split('\t')
                if l[0] in self.words and l[1] in self.words:
                    orthography_original_similarities[l[0]][l[1]] = float(l[2])
                    orthography_original_similarities[l[1]][l[0]] = float(l[2])

        # Turning defaultdict into a regular dict
        orthography_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in orthography_original_similarities.items()}

        return orthography_original_similarities

    def get_visual(self):

        visual_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'visual', 'visual.sims'), 'r') as visual_original_file:
            for i, l in enumerate(visual_original_file):
                if i > 0:
                    l = l.strip().split('\t')
                    if l[0] in self.words and l[1] in self.words:
                        visual_original_similarities[l[0]][l[1]] = float(l[3])
                        visual_original_similarities[l[1]][l[0]] = float(l[3])

        # Turning defaultdict into a regular dict
        visual_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in visual_original_similarities.items()}

        return visual_original_similarities

    def get_wordnet(self):

        wordnet_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        with open(os.path.join('computational_models', 'wordnet', 'wordnet.sims'), 'r') as wordnet_original_file:
            for i, l in enumerate(wordnet_original_file):
                l = l.strip().split('\t')
                if l[0] in self.words and l[1] in self.words:
                    wordnet_original_similarities[l[0]][l[1]] = float(l[2])
                    wordnet_original_similarities[l[1]][l[0]] = float(l[2])

        # Turning defaultdict into a regular dict
        wordnet_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in wordnet_original_similarities.items()}

        return wordnet_original_similarities

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

    def get_new_cooc(self, mode='basic_cooc'):

        cooc_original_similarities = collections.defaultdict(lambda : collections.defaultdict(float))
        mode_indices = {'basic_cooc' : 1, 'ppmi' : 3, 'w2v_style' : 5}

        base_folder = os.path.join('computational_models', 'cooc', 'new_cooc')
        for w in self.words:
            with open(os.path.join(base_folder, '{}.cooc'.format(w))) as w_file:
                lines = [l.strip().split('\t') for l in w_file.readlines()]
            header = lines[0]
            #print([(i, h) for i, h in enumerate(header)])
            sims = lines[1:]
            for l in sims:
                if l[0] != w:
                    cooc_original_similarities[w][l[0]] = float(l[mode_indices[mode]])

        # Turning defaultdict into a regular dict
        cooc_original_similarities = {k_one : {k_two : v_two for k_two, v_two in v_one.items()} for k_one, v_one in cooc_original_similarities.items()}

        return cooc_original_similarities

    def compare_models(self, tsne=False):
       
        words = [c for c in itertools.combinations(self.words, 2)]
        models = dict()
        sim_models = collections.defaultdict(list)
        results = dict()

        models['w2v'] = self.get_w2v_sims()
        models['original_cooc'] = self.get_original_cooc()
        models['wordnet'] = self.get_wordnet()
        models['new_cooc'] = self.get_new_cooc()
        models['ppmi'] = self.get_new_cooc(mode='ppmi')
        models['orthography'] = self.get_orthography()
        #models['w2v_window_cooc'] = self.get_new_cooc(mode='w2v_style')

        for m_name, m in models.items():
            for w_one, w_two in words:
                sim_models[m_name].append(m[w_one][w_two])
        
        combs = itertools.combinations([k for k in models.keys()], 2)
        for m_one, m_two in combs:

            model_one = sim_models[m_one]
            model_two = sim_models[m_two]

            pearson = stats.pearsonr(model_one, model_two)[0]
            spearman = stats.spearmanr(model_one, model_two)[0]

            results[(m_one, m_two)] = ['pearson: {}'.format(pearson), 'spearman: {}'.format(spearman)]
            results[(m_two, m_one)] = ['pearson: {}'.format(pearson), 'spearman: {}'.format(spearman)]

        if tsne == True:
 
            out_folder = 'comp_models_visualization'
            os.makedirs(out_folder, exist_ok=True)

            ### TSNE plot
            tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
            embeddings = tsne_model_en_2d.fit_transform([v for k, v in sim_models.items()])

            fig, ax = pyplot.subplots()
            
            for m_name, emb in zip([k for k in sim_models.keys()], embeddings):
                ax.scatter(emb[0], emb[1], label=m_name)

            ax.set_title( 'T-SNE visualizations of different models', fontsize='xx-large', fontweight='bold', pad = 15.0)
            ax.legend()

            pyplot.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
            pyplot.savefig(os.path.join(out_folder, 'tsne_computational_models_comparison'), format='png', bbox_inches='tight', dpi=300)

            ### Confusion matrix
            labels = [k for k in models.keys()]
            for l in labels:
                results[(l, l)] = [': 1.', ': 1.']

            for score_index in range(2):
                score = 'pearson' if score_index==0 else 'spearman'
                matrix = [[float(results[(l_one, l_two)][score_index].split(': ')[1]) for l_two in labels] for l_one in labels]
                confusion_matrix(matrix, labels, 'computational_models', score, '{}/'.format(out_folder))

        return results

### Collects and organizes the evoked responses

class EvokedResponses:

    def __init__(self, subject_number):

        self.folder = '/import/cogsci/andrea/dataset/neuroscience/conscious_unconscious_processing/preprocessed_files/sub-{:02}'.format(subject_number)
        #self.folder = '/mnt/c/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing/preprocessed_files/sub-{:02}'.format(subject_number)
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
                    # 0 = word, 1 = target/filler, 2 = accuracy, 3 = awareness, 5 = trial good or not, 
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

