import matplotlib
import numpy
import os

from matplotlib import pyplot
from scipy import stats
numpy.seterr(all='raise')

plot_path = os.path.join('plots', 'two', 'behavioural')
os.makedirs(plot_path, exist_ok=True)

file_path = os.path.join('results', 'behavioural', 'two', 
                                      'experiment_two.csv')
with open(file_path) as i:
    lines = [l.strip().split(',') for l in i.readlines()]

headers = lines[0]
data = lines[1:]
for d in data:
    if len(d) < len(headers):
        data.insert(0, '')

res_dict = {k : [d[k_i] for d in data] for k_i, k in enumerate(headers)}
subjects = sorted(list(set([int(w) for w in res_dict['subject']])))
mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}

### Accuracy scores

all_correct = list()
all_wrong = list()

for s in subjects:
    s_data = [w for w_i, w in enumerate(res_dict['Accuracy']) if int(res_dict['subject'][w_i])==s]

    correct = len([w for w in s_data if w == 'correct'])
    wrong = len([w for w in s_data if w == 'wrong'])
    total = correct+wrong

    all_correct.append(correct/total)
    all_wrong.append(wrong/total)

all_correct = numpy.array(all_correct)
all_wrong = numpy.array(all_wrong)

fig, ax = pyplot.subplots()

ax.bar(subjects, all_correct, label='correct')
ax.bar(subjects, all_wrong, bottom=all_correct, label='wrong')

ax.set_xticks(subjects)

title = 'Overall performances for each subject'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=2)
pyplot.tight_layout()
#pyplot.show()
file_path = os.path.join(plot_path, 'overall_performance.png')
pyplot.savefig(file_path)

pyplot.clf()

### Awareness scores

all_low = list()
all_medium = list()
all_high = list()

for s in subjects:
    s_data = [w for w_i, w in enumerate(res_dict['PAS score']) if int(res_dict['subject'][w_i])==s]

    low = len([w for w in s_data if int(w) == 1])
    medium = len([w for w in s_data if int(w) == 2])
    high = len([w for w in s_data if int(w) == 3])

    total = low + medium + high

    all_low.append(low/total)
    all_medium.append(medium/total)
    all_high.append(high/total)

all_low = numpy.array(all_low)
all_medium = numpy.array(all_medium)
all_high = numpy.array(all_high)

fig, ax = pyplot.subplots()

ax.bar(subjects, all_low, label='low')
ax.bar(subjects, all_medium, bottom=all_low, label='medium')
ax.bar(subjects, all_high, bottom=all_medium+all_low, label='high')
ax.set_xticks(subjects)

title = 'PAS scores for each subject'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()
#pyplot.show()
file_path = os.path.join(plot_path, 'pas.png')
pyplot.savefig(file_path)

pyplot.clf()

### Accuracy for PAS

all_low = list()
all_medium = list()
all_high = list()

pas = [1,2,3]
acc = ['correct', 'wrong']

results = {p : list() for p in pas}

for s in subjects:
    s_data = [(int(w), res_dict['Accuracy'][w_i]) for w_i, w in enumerate(res_dict['PAS score']) if int(res_dict['subject'][w_i])==s]
    for p in pas:
        current_corr = len([1 for s_p, s_a in s_data if s_p==p and s_a=='correct'])
        current_wrong = len([1 for s_p, s_a in s_data if s_p==p and s_a=='wrong'])
        if current_corr == 0 and current_wrong == 0:
            s_r = numpy.nan
        else:
            s_r = current_corr/(current_corr+current_wrong)
        results[p].append(s_r)

fig, ax = pyplot.subplots()

#ax.violinplot([p_r for k, p_r in results.items()])
for k_p_r_i, k_p_r in enumerate(results.items()):
    ax.scatter([i+1+k_p_r_i*0.1 for i in range(len(k_p_r[1]))], k_p_r[1], \
                            label=mapper[k_p_r[0]])
ax.legend()
ax.hlines(y=0.5, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.hlines(y=1., xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.set_xticks([s+0.1 for s in subjects])
ax.set_xticklabels(subjects)
title = 'Accuracy scores across subjects'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()

file_path = os.path.join(plot_path, 'accuracies.png')
pyplot.savefig(file_path)
#pyplot.show()

pyplot.clf()

## D-prime

pas = [1,2,3]
acc = ['correct', 'wrong']

results = {p : list() for p in pas}

for s in subjects:
    s_data = [(int(w), res_dict['Accuracy'][w_i], res_dict['required_answer'][w_i]) for w_i, w in enumerate(res_dict['PAS score']) if int(res_dict['subject'][w_i])==s]
    for p in pas:
        current_corr = len([1 for s_p, s_a, req in s_data if s_p==p and s_a=='correct' and req=='YES']) / len([1 for s_p, s_a, req in s_data if req=='YES'])
        current_wrong = len([1 for s_p, s_a, req in s_data if s_p==p and s_a=='wrong' and req=='NO']) / len([1 for s_p, s_a, req in s_data if req=='NO'])
        z_hit = stats.norm.ppf(current_corr)
        z_fa = stats.norm.ppf(current_wrong)
        try:
            d_prime = z_hit - z_fa
        except FloatingPointError:
            d_prime = numpy.nan
        #print([p, d_prime])
        results[p].append(d_prime)

fig, ax = pyplot.subplots()

#ax.violinplot([p_r for k, p_r in results.items()])
for k_p_r_i, k_p_r in enumerate(results.items()):
    ax.scatter([i+1+k_p_r_i*0.1 for i in range(len(k_p_r[1]))], k_p_r[1], \
                            label=mapper[k_p_r[0]])
ax.legend()
ax.hlines(y=0.0, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.hlines(y=1.0, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.set_xticks([s+0.1 for s in subjects])
ax.set_xticklabels(subjects)
title = 'D-prime scores across subjects'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()

file_path = os.path.join(plot_path, 'dprime.png')
pyplot.savefig(file_path)
#pyplot.show()

pyplot.clf()
