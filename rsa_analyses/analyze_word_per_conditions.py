import argparse
import numpy
import collections
import copy
import os

from matplotlib import pyplot, cm

from io_utils import EvokedResponses
from rsa_utils import restrict_evoked_responses

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=str, default='objective_accuracy')
parser.add_argument('--word_selection', type=str, default='all_words')
args = parser.parse_args()

base_folder = 'word_analyses_plots'
os.makedirs(base_folder, exist_ok=True)

analyses = ['objective_accuracy', 'subjective_judgments']
mapping = {'objective_accuracy' : ['correct', 'wrong'], \
           'subjective_judgments' : ['high', 'medium', 'low'], \
          }

word_selection = ['all_words', 'targets_only']
erps = [1,2,4,8]
cmap = copy.copy(cm.get_cmap("RdYlGn"))
cmap.set_under(color='white')

for w in word_selection:
    words_per_condition = collections.defaultdict(lambda : collections.defaultdict(list))

    args.word_selection = w
    vmax = 40. if 'all' in w else 20.

    for a in analyses:
        args.analysis = a
        for s in range(2, 18):
            e = EvokedResponses(s)
            import pdb; pdb.set_trace()
            e = restrict_evoked_responses(args, e)
            for n in erps:
                n_per_sub = {k : len([w for w in v.items() if len(w[1]) >= n]) for k, v in e.items()}
                for k, l_w in n_per_sub.items():
                    words_per_condition[k][n].append(l_w)

    fig, ax = pyplot.subplots(1, 2, constrained_layout=True, \
                              figsize=(10., 5.), \
                             )
    fig.suptitle('Number of words available in each data split considering {}'.format(w.replace('_', ' ')))
    for i, a in enumerate(analyses):

        title = 'Accuracy scores' if i==0 else 'Subjective reports'
        ax[i].set_title(title)

        matrix = list()
        labels = list()

        for c in mapping[a]:

            for n, n_list in words_per_condition[c].items():
                matrix.append(n_list)
                if n == 1:
                    lab = '{}   min. {}'.format(c, n)
                else:
                    lab = 'min. {}'.format(n)
                labels.append(lab)
          
            matrix.append([-1 for i in range(15)])
            labels.append('')

        ax[i].imshow(matrix, cmap=cmap, extent=(0,len([i for i in range(15)]),len(matrix),0), vmin=0., vmax=vmax, aspect='auto')
        #ax[i].set_xticks([i+.5 for i in range(15)])
        ax[i].set_yticks([i+.5 for i in range(len(labels))])

        ax[i].set_yticklabels(labels, fontsize='xx-small')

        mid_values = [10, 30] if 'all' in w else [5, 15]
        for l in range(len(labels)):
            for j in range(15):
                value = matrix[l][j]
                
                if value <= mid_values[0] or value >= mid_values[1]:
                    color = 'white'
                else:
                    color = 'gray'
                ax[i].text(j+.5, l+.5, value, ha="center", va="center", color=color)
        ax[i].hlines(y=[i for i in range(len(labels))], xmin=0, xmax=len([i for i in range(15)]), color='white', linewidths=1.)
        ax[i].vlines(x=[i for i in range(15)], ymin=0, ymax=len(labels), color='white', linewidths=1.)
        ax[i].set_xlabel('subjects')

    pyplot.savefig(os.path.join(base_folder, 'word_availability_matrix_{}.png'.format(w)), dpi=600)
    pyplot.clf()

'''
with open('word_availability_per_condition.txt', 'w') as o:
    for cond, cond_dict in words_per_condition.items():
        o.write('Data availability for {}\n\n'.format(cond.replace('_', ' ').capitalize()))
        for n, n_list in cond_dict.items():
            o.write('\tUsing {} evoked responses: {}\n'.format(n, n_list))
        o.write('\n\n\n')
'''
