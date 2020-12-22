import mne
import os
import collections
import itertools
import numpy
import scipy
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

event_mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}

w2v_list = list()
for i in [50000, 100000, 150000]:
    w2v_similarities = collections.defaultdict(float)
    with open(os.path.join('computational_models', 'sims_w2v_vocab_{}'.format(i)), 'r') as w2v_file:
        for l in w2v_file:
            l = l.strip().split('\t')
            w2v_similarities[(l[0], l[1])] = float(l[3])
    w2v_list.append(w2v_similarities)

for s in range(3, 12):
    
    events = list()
    events_selector = collections.defaultdict(list)

    folder = '/mnt/c/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing/sub-{:02}'.format(s)
    epochs = mne.read_epochs(os.path.join(folder, 'sub-{:02}_highpass-100Hz-epoched-concatenated.fif'.format(s)))
    with open(os.path.join(folder, 'sub-{:02}_events_rejected_or_good.txt'.format(s))) as f:
        for l in f:
            l = l.strip().split('\t')
            if l[5] != 'rejected':
                events.append(l)

    assert len(events) == len(epochs)

    for event_index, e in enumerate(events):
        if e[1] == 'target' and e[3] == 'correct':
            events_selector[event_mapper[int(e[4])]].append(event_index)

    epochs = epochs.get_data()

    sampling_points = epochs.shape[-1]
    time_step = (sampling_points/204.8)/sampling_points
    time_points = collections.defaultdict(float)

    for i in range(sampling_points):
        if i == 0:
            time_points[i] = -0.2
        else:
            current_time = time_points[i-1] + time_step
            if current_time <= 1.:
                time_points[i] = current_time
            else:
                break

    feature_standardizer = mne.decoding.Scaler(scalings='mean')
    epochs = feature_standardizer.fit_transform(epochs)

    subject_results = collections.defaultdict(list)

    for certainty_level, indices in events_selector.items():
        current_evoked_eeg = collections.defaultdict(list)
        for index in indices:
            current_event = events[index] 
            current_word = current_event[0]
            current_evoked_eeg[current_word].append(epochs[index])
        current_evoked_eeg = {k : numpy.average(v, axis=0) for k, v in current_evoked_eeg.items()}

        word_combs = [k for k in itertools.combinations(current_evoked_eeg.keys(), r=2)]

        eeg_similarities = collections.defaultdict(list)
        w2v_ordered = list()
        for word_one, word_two in word_combs:
            w2v_ordered.append(w2v_list[2][(word_one, word_two)])
            for sample_point, time_point in time_points.items():
                eeg_one = current_evoked_eeg[word_one][:, sample_point]
                eeg_two = current_evoked_eeg[word_two][:, sample_point]
                eeg_similarities[time_point].append(scipy.stats.pearsonr(eeg_one, eeg_two)[0])

        for time_point, eeg_time_point in eeg_similarities.items():
            correlation_with_computational = scipy.stats.pearsonr(eeg_time_point, w2v_ordered)[0]
            subject_results[certainty_level].append(correlation_with_computational)
        subject_results[certainty_level].append('n_words={}'.format(len(current_evoked_eeg.keys())))
        
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)
    for certainty, results in subject_results.items():
        plot_results = results[:-1]
        number_words = results[-1]
        ax.plot([v for k, v in time_points.items()], plot_results, label='{} ({})'.format(certainty, number_words))
    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Time')
    ax.set_title('Correlation with Word2Vec at each time point for subject {}'.format(s), pad=40)
    plt.savefig(os.path.join('temp', 'rsa_w2v_sub_{:02}.png'.format(s)))
    plt.clf()
    plt.close()
