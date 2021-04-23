import mne
import os
import pandas
import tqdm
import matplotlib
import argparse

import numpy as np

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject import Ransac, get_rejection_threshold, AutoReject, compute_thresholds

from autoreject.utils import interpolate_bads

#matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, \
                    required=True, help='Folder where \
                    the original data is stored')
args = parser.parse_args()

### Channel naming

eeg_channels = ['{}{}'.format(letter, number) for number in range(1, 33) for letter in ['A', 'B', 'C', 'D']]
eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

montage = mne.channels.make_standard_montage(kind='biosemi128')
random_state = 1

number_of_subjects = len(os.listdir(args.data_folder))

for s in range(1, number_of_subjects+1):

    epochs_list = list()

    subject_folder = 'sub-{:02}'.format(s)
    eeg_folder = os.path.join(subject_folder, 'sub-{:02}_eeg'.format(s))
    events_folder = os.path.join(subject_folder, 'sub-{:02}_events'.format(s))

    output_folder = os.path.join('preprocessed_data', subject_folder)
    os.makedirs(os.path.join(output_folder, exist_ok=True)

    for r in range (1, 33):

        ### Loading run raw data
        run_path = os.path.join(eeg_folder, 'sub-{:02}_run-{:02}.bdf'.format(s, r))

        raw_raw = mne.io.read_raw_bdf(run_path, preload=True, \
                                      eog=eog_channels, \
                                      exclude=excluded_channels, \
                                      verbose=False)
        raw_raw.set_montage(montage)

        ### Extracting events from file to a dictionary
        events_path = os.path.join(events_folder, \
                                   'run-{:02}.events'.format(r))
        with open(events_path) as events_file:
            lines = [l.strip().split('\t') for l in events_file]
        header = lines[0]
        data = lines[1:]

        events_dict = {h : [l[h_i] for l in data] for h_i, h in enumerate(header)}

        current_events = mne.find_events(raw_raw)
        if len(current_events.shape[0]) != 20:
            print('sub {} r {}'.format(s, r))
            
        ### Cropping so as to remove useless recorded samples before/after testing
        sampling = raw_raw.info['sfreq']
        minimum_t = max((current_events[0][0] / sampling) -0.5, 0)
        maximum_t = min((current_events[-1][0] / sampling) + 1.5, \
                         raw_raw.n_times)
        cropped_raw = raw_raw.crop(tmin=minimum_t, tmax=maximum_t)
        cropped_events = mne.find_events(cropped_raw)
        assert cropped_events.shape[0] == current_events.shape[0]

        ### Band pass
        ### Low-pass at 80 Hz
        cropped_raw.filter(None, 80)

        # Epoch the data
        print('Epoching')
        epochs = mne.Epochs(raw=cropped_raw, \
                            events=events, \
                            event_id=[int(i) if s != 2 else int(i)-10 \
                                      for i in events_dict['Trigger code']], \
                            tmin=-0.1, tmax=1.1, \
                            picks=picks, \ 
                            preload=True, decim=8)

        ### Trigger code correction for subject 1
        if s == 2:
            epochs.events = numpy.array([l+[0,0,10] for l in epochs.events])

        # Finding bad channels

        bad_channels_finder = AutoReject(random_state=random_state, \
                                         n_jobs=os.cpu_count())
        epochs, autoreject_log = bad_channels_finder.fit_transform(epochs, \
                                                            return_log=True)

        trials = {i : 'good' for i in range(len(autoreject_log.bad_epochs))}
        for epoch_index, bad_epoch in enumerate(autoreject_log.bad_epochs):
            if bad_epoch:
                trials[epoch_index] = 'rejected'


        ### ICA

        ### High-pass at 1Hz to get reasonable thresholds in autoreject
        hipass_raw = subject_raw.copy().drop_channels(['EXG1', 'EXG2'])
        hipass_raw = hipass_raw.filter(1., None)

        print('Now moving to ICA estimation...')

        ica = ICA(random_state=1, n_components=0.99)
        ica.fit(hipass_raw)
        del hipass_raw

        ### EOG detection

        eog_epochs = create_eog_epochs(hipass_raw, \
                                       tmin=-.5, tmax=.5, \
                                       preload=True)
        eog_epochs.decimate(5)
        eog_epochs.load_data()
        eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)

        # removing Individual Components
        print('Removing Indipendent Components')
        ica.exclude = []
        ica.exclude.extend(eog_inds)

        epochs.load_data()
        ica.apply(epochs)

        print('Getting rejection thresholds by autoreject')
        reject = get_rejection_threshold(epochs.copy().crop(None, tmax=1.1), \
                                         random_state=random_state)
        epochs.drop_bad(reject=reject)

        ### Obtaining the drop log so as to filter original events
        used_indiced = list()
        drop_log = epochs.drop_log
        assert len(drop_log) == len(data)
        for ep_i, ep in enumerate(drop_log):
            if len(ep) > 0:
                pass
            else:
                used_indices.append(ep_i)

        mne.set_eeg_reference(epochs, ref_channels='average', \
                              ch_type='eeg')
        epochs_list.append(epochs) 

    print('Writing epochs to file')
    all_epochs = mne.concatenate_epochs(epochs_list)
    epochs_file = 'sub-{:02}_epo.fif'.format(s)
    epochs.save(os.path.join(output_folder, epochs_file))


    '''
    subject_raw, events = mne.concatenate_raws(raws_list, events_list=events_list)
    all_events_ids = {k : v for k, v in all_events_ids.items() if v in [i[2] for i in events]}
    assert len(original_events_list) == len(events)

    subject_raw.filter(0.1, None, fir_design='firwin')


    ### Epoching

    ###############################################################################
    # We define the events and the onset and offset of the epochs

    tmin = -0.2
    tmax = 1.5  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
    reject_tmax = 1.0  # duration we really care about

    baseline = (-0.2, 0)


    ### projection=True applies the reference from average directly

    picks = mne.pick_types(subject_raw.info, eeg=True, stim=True, eog=False, exclude=())





    with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
        print('Found {} EOG indices'.format(len(eog_inds)))
        o.write('Found {} EOG indices\n\n'.format(len(eog_inds)))


    
    print('Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))

    with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
        o.write('  Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))
        for epoch_index, tup in enumerate(epochs.drop_log):
            o.write('event {}\trejection reason: {}\n'.format(epoch_index, tup))

    ### Merging original events log with their 'good' or 'rejected' status

    rejected_epochs_eog = [i for i, k in enumerate(epochs.drop_log) if len(k) != 0]

    c = 0
    for epoch_index, kept_or_not in enumerate(trials.keys()):
        if kept_or_not == 'good':
            if c in rejected_epochs_eog:
                trials[c] = 'rejected_eog'
            else:
                pass
            c += 1

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

'''
