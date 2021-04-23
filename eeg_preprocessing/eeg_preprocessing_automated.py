import mne
import os
import pandas
import tqdm
import matplotlib
matplotlib.use('Agg')

import numpy as np

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject import Ransac, get_rejection_threshold, AutoReject, compute_thresholds

from autoreject.utils import interpolate_bads

### Channel naming

eeg_channels = ['{}{}'.format(letter, number) for number in range(1, 33) for letter in ['A', 'B', 'C', 'D']]
eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

montage = mne.channels.make_standard_montage(kind='biosemi128')

### load the stimulus set and translate them into events
Stimuli = pandas.read_csv('stimuli_final.csv', delimiter=';')
trial_stimuli = [k for k in range(30)] + [k for k in range(40, 50)]
all_events_ids = {Stimuli['word'][stim] : stim+10 for stim in trial_stimuli}

main_folder = '/mnt/c/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing'

#for s in tqdm(range(3, 18)):
#for s in tqdm(range(12, 18)):
for s in [2]:

    raws_list = list()
    epochs_list = list()
    original_events_list = list()
    events_list = list()

    eeg_folder = 'subject{}/sub-{:02}_eeg'.format(s, s)
    events_folder = 'subject{}/sub-{:02}_events'.format(s, s)
    output_folder = 'sub-{:02}'.format(s)

    os.makedirs(os.path.join(main_folder, output_folder), exist_ok=True)

    for r in range (1, 33):
        ### Small fix for subject 2
        if s == 2:
            r = r - 1

        raw_raw = mne.io.read_raw_bdf(os.path.join(main_folder, eeg_folder, 'Testdata{}.bdf'.format(r)), preload=True, eog=eog_channels, exclude=excluded_channels)
        raw_raw.set_montage(montage)
        current_events = mne.find_events(raw_raw)
        sampling = raw_raw.info['sfreq']

        cropped_raw = raw_raw.crop(tmin=max((current_events[0][0]/sampling)-0.5, 0), tmax=(current_events[-1][0]/sampling)+1.5)

        events_raw = pandas.read_csv(os.path.join(main_folder, events_folder, 'run_{:02}_events_log.csv'.format(r)))
        ### Small fix for subject 2
        if r == 2:
            events_info = [(events_raw['Word'][index], events_raw['Group'][index], events_raw['Trigger code'][index]+10, events_raw['Prediction outcome'][index], events_raw['Certainty'][index]) for index in range(20)]
        else:
            events_info = [(events_raw['Word'][index], events_raw['Group'][index], events_raw['Trigger code'][index], events_raw['Prediction outcome'][index], events_raw['Certainty'][index]) for index in range(20)]
        original_events_list.extend(events_info)
        events_list.append(current_events)
        raws_list.append(cropped_raw)

    subject_raw, events = mne.concatenate_raws(raws_list, events_list=events_list)
    all_events_ids = {k : v for k, v in all_events_ids.items() if v in [i[2] for i in events]}
    assert len(original_events_list) == len(events)

    ### Band pass
    ### Low-pass at 40 Hz
    subject_raw.filter(0.1, None, fir_design='firwin')
    subject_raw.filter(None, 40, fir_design='firwin')


    ### Epoching

    ###############################################################################
    # We define the events and the onset and offset of the epochs

    tmin = -0.2
    tmax = 1.5  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
    reject_tmax = 1.0  # duration we really care about
    random_state = 1

    baseline = (-0.2, 0)


    ### projection=True applies the reference from average directly

    picks = mne.pick_types(subject_raw.info, eeg=True, stim=True, eog=False, exclude=())

    # Epoch the data
    print('  Epoching')
    epochs = mne.Epochs(raw=subject_raw, events=events, event_id=all_events_ids, tmin=tmin, tmax=tmax, picks=picks, reject_tmax=reject_tmax, preload=True, decim=10)

    # Finding bad channels

    #ransac = Ransac(verbose='progressbar', picks=['eeg'], n_jobs=1)
    #epochs_clean = ransac.fit_transform(epochs)
    bad_channels_finder = AutoReject(random_state=random_state, n_jobs=6, verbose='tqdm')
    epochs, autoreject_log = bad_channels_finder.fit_transform(epochs, return_log=True)

    trials = {i : 'good' for i in range(len(autoreject_log.bad_epochs))}
    for epoch_index, bad_epoch in enumerate(autoreject_log.bad_epochs):
        if bad_epoch:
            trials[epoch_index] = 'rejected_channel'

    ### ICA

    ### High-pass at 1Hz to get reasonable thresholds in autoreject
    hipass_raw = subject_raw.copy().drop_channels(['EXG1', 'EXG2'])
    hipass_raw = hipass_raw.filter(1., None)

    print('Now moving to ICA estimation...')

    ica = ICA(random_state=1, n_components=0.999)
    ica.fit(hipass_raw)
    del hipass_raw

    ### EOG detection

    eog_epochs = create_eog_epochs(subject_raw, tmin=-.5, tmax=.5, preload=False)
    eog_epochs.decimate(5)
    eog_epochs.load_data()
    eog_epochs.apply_baseline((None, None))
    eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)

    with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
        print('Found {} EOG indices'.format(len(eog_inds)))
        o.write('Found {} EOG indices\n\n'.format(len(eog_inds)))

    # removing Individual Components
    print('Removing Indipendent Components')
    ica.exclude = []
    ica.exclude.extend(eog_inds)

    epochs.load_data()
    ica.apply(epochs)

    print('Getting rejection thresholds by autoreject')
    reject = get_rejection_threshold(epochs.copy().crop(None, tmax=reject_tmax), random_state=random_state)
    epochs.drop_bad(reject=reject)

    mne.set_eeg_reference(epochs, ref_channels='average', projection=True, ch_type='eeg')
    
    print('Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))

    with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
        o.write('  Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))
        for epoch_index, tup in enumerate(epochs.drop_log):
            o.write('event {}\trejection reason: {}\n'.format(epoch_index, tup))

    ### Merging original events log with their 'good' or 'rejected' status

    '''
    rejected_epochs_eog = [i for i, k in enumerate(epochs.drop_log) if len(k) != 0]

    c = 0
    for epoch_index, kept_or_not in enumerate(trials.keys()):
        if kept_or_not == 'good':
            if c in rejected_epochs_eog:
                trials[c] = 'rejected_eog'
            else:
                pass
            c += 1
    '''

    with open(os.path.join(main_folder, output_folder, 'sub-{:02}_events_rejected_or_good.txt'.format(s)), 'a') as o:
        for epoch_index, result_tuple in enumerate(epochs.drop_log):
            for element in original_events_list[epoch_index]:
                o.write('{}\t'.format(element))
            if len(result_tuple) == 0:
                result = 'good'
            else:
                result = 'rejected'
            o.write('{}\n'.format(result))

    del subject_raw, ica, eog_epochs


    print('Writing to file')
    epochs.save(os.path.join(main_folder, output_folder, 'sub-{:02}_highpass-{}Hz-epoched-concatenated.fif'.format(s, 100)))
