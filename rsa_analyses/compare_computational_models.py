import collections
import itertools
import numpy
import os

from io_utils import ComputationalModels

from sklearn.manifold import TSNE
from scipy import stats
from matplotlib import pyplot

import sys
sys.path.append('/import/cogsci/andrea/github/')

from grano.plot_utils import confusion_matrix

def compare_models(AllModels, tsne=False):
   
    ### Loading only target words
    word_selections = dict()
    
    targets = list()
    with open('../lab_experiment/stimuli_final.csv', 'r') as stimuli_file:
        for i, l in enumerate(stimuli_file):
            if i > 0: 
                l = l.strip().split(';')
                if l[2] == 'target':
                    targets.append(l[0])

    word_selections = {'target_words' : targets, 'all_words' : AllModels.words}

    models = dict()
    models['w2v'] = AllModels.get_w2v_sims()
    models['original_cooc'] = AllModels.get_original_cooc()
    models['wordnet'] = AllModels.get_wordnet()
    models['new_cooc'] = AllModels.get_new_cooc()
    models['ppmi'] = AllModels.get_new_cooc(mode='ppmi')
    models['cslb'] = AllModels.get_cslb()
    models['orthography'] = AllModels.get_orthography()
    models['visual'] = AllModels.get_visual()
    models['CORnet'] = AllModels.get_CORnet()
    #models['w2v_window_cooc'] = AllModels.get_new_cooc(mode='w2v_style')

    for selection, chosen_words in word_selections.items():

        print('Now comparing {}...'.format(selection))
        words = [c for c in itertools.combinations(chosen_words, 2)]
        sim_models = collections.defaultdict(list)
        results = dict()

        for m_name, m in models.items():
            for w_one, w_two in words:
                try: 
                    sim_models[m_name].append(m[w_one][w_two])
                except KeyError:
                    import pdb; pdb.set_trace()
        
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

            ax.set_title( 'T-SNE visualizations of different models considering {}'.format(selection), fontsize='xx-large', fontweight='bold', pad = 15.0)
            ax.legend()

            pyplot.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
            pyplot.savefig(os.path.join(out_folder, '{}_tsne_computational_models_comparison.png'.format(selection)), format='png', bbox_inches='tight', dpi=300)
            pyplot.clf()

            ### Confusion matrix
            labels = [k for k in models.keys()]
            for l in labels:
                results[(l, l)] = [': 1.', ': 1.']

            for score_index in range(2):
                score = 'pearson' if score_index==0 else 'spearman'
                matrix = [[float(results[(l_one, l_two)][score_index].split(': ')[1]) for l_two in labels] for l_one in labels]
                confusion_matrix(matrix, labels, 'computational_models', '{} considering {}'.format(score, selection), '{}/{}_'.format(out_folder, selection))

    return results

AllModels = ComputationalModels()
compare_models(AllModels, tsne=True)
