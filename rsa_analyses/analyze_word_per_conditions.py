import argparse
import numpy
import collections

from matplotlib import pyplot

from io_utils import EvokedResponses
from rsa_utils import restrict_evoked_responses

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=str, default='objective_accuracy')
parser.add_argument('--word_selection', type=str, default='all_words')
args = parser.parse_args()

analyses = ['objective_accuracy', 'subjective_judgments']
words_per_condition = collections.defaultdict(list)

for a in analyses:
    args.analysis = a
    for s in range(2, 4):
        e = EvokedResponses(s)
        e = restrict_evoked_responses(args, e)
        cond_words = {k : len(v) for k,v in e.items()}
        for k, l_w in cond_words.items():
            words_per_condition[k].append(l_w)

fig, ax = pyplot.subplots()
x = [i+2 for i in range(len(words_per_condition['wrong']))]
c = 0
for k, v in words_per_condition.items():

    ax.bar(x, v, width=0.4, label=k)
    c += .5
    x = numpy.add(x, c)
pyplot.savefig('prova.png')
