import matplotlib
import mne
import numpy
import os

from matplotlib import pyplot
from scipy import stats

data_path = 'classification_per_subject'
plot_path = 'new_plots'
modes = ['correct', 'wrong', \
         'high', 'medium', 'low']

os.makedirs(plot_path, exist_ok=True)
subjects = len(os.listdir(data_path))

for m in modes:
    scores = list()
    missing = list()
    for s in range(1, subjects+1):
        sub_path = os.path.join(data_path, 'sub-{:02}'.format(s), \
                                'sub_{:02}_{}_scores.txt'.format(s, m))
        if not os.path.exists(sub_path):
            missing.append('sub {} mode {}'.format(s, m))
        else:
            with open(sub_path) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            times = numpy.array(lines[0], dtype=numpy.single)
            sub_scores = numpy.array(lines[1], dtype=numpy.single)
            scores.append(sub_scores)

    average_scores = numpy.average(scores, axis=0)
    p_values = stats.ttest_1samp(numpy.array(scores), popmean=0.5, alternative='greater', \
                                 axis=0).pvalue
    corrected_p = mne.stats.fdr_correction(p_values)[1]
    significant_indices = [p_i for p_i, p in enumerate(corrected_p) if p<=0.05]
    assert p_values.shape == times.shape
    assert average_scores.shape == times.shape

    fig, ax = pyplot.subplots(figsize=(12,5))
    ax.plot(times, average_scores, \
            label='N={}'.format(len(scores)))
    for p in scores:
        ax.scatter(times, p, \
                   alpha=0.1, s=4)
    for p in significant_indices:
        ax.scatter(times[p], average_scores[p], color='white', \
                   edgecolors='black')
    ax.legend()
    ax.hlines(y=0.5, xmin=times[0], xmax=times[-1], \
              color='gray', linestyle='dashed')
    ax.set_title('Classification scores for {}'.format(m))
    pyplot.savefig(os.path.join(plot_path, 'classification_{}.png'.format(m)), dpi=300)

