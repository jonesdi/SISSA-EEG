import mne
import os
import pandas
import tqdm
import matplotlib
import argparse
import numpy
import autoreject

import numpy as np

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject.utils import interpolate_bads

#matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, \
                    required=True, help='Folder where \
                    the original data is stored')
args = parser.parse_args()

### Channel naming

eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

montage = mne.channels.make_standard_montage(kind='biosemi128')
random_state = 1

number_of_subjects = len(os.listdir(args.data_folder))

for s in range(1, number_of_subjects+1):

    epochs_list = list()
    present_events = list()
    all_events = dict()

    subject_folder = 'sub-{:02}'.format(s)
    eeg_folder = os.path.join(args.data_folder, subject_folder, 'sub-{:02}_eeg'.format(s))
    events_folder = os.path.join(args.data_folder, subject_folder, 'sub-{:02}_events'.format(s))

    output_folder = os.path.join('preprocessed_data', subject_folder)
    os.makedirs(os.path.join(output_folder), exist_ok=True)

    for r in range(1, 33):

        ### Loading run raw data
        run_path = os.path.join(eeg_folder, 'sub-{:02}_run-{:02}.bdf'.format(s, r))
        assert os.path.exists(run_path)

        raw_raw = mne.io.read_raw_bdf(run_path, preload=True, \
                                      eog=eog_channels, \
                                      exclude=excluded_channels, \
                                      verbose=False)
        raw_raw.set_montage(montage)

        ### Extracting events from file
        events_path = os.path.join(events_folder, \
                                   'run-{:02}.events'.format(r))
        with open(events_path) as events_file:
            lines = [l.strip().split('\t') for l in events_file]
        header = lines[0]
        data = lines[1:]

        current_events = mne.find_events(raw_raw)
        if current_events.shape[0] != 20:
            print('sub {} r {}'.format(s, r))
        if s == 1:
            events_from_file = [d for d in data if int(d[3]) != 10]
        else:
            events_from_file = data.copy()

        try:
            assert len(events_from_file) == current_events.shape[0]
        except AssertionError:
            assert len(events_from_file) > current_events.shape[0]
            old_events = events_from_file.copy()
            events_from_file = list()
            to_be_subtracted = 0
            for e_i, e in enumerate(old_events):
                index = max(0, e_i - to_be_subtracted)
                trigger = int(e[3]) if s!=1 else int(e[3])-10
                if trigger == current_events[index, 2]:
                    events_from_file.append(e)
                else:
                    to_be_subtracted += 1
        assert len(events_from_file) == current_events.shape[0]
            
        ### Cropping so as to remove useless recorded samples before/after testing
        sampling = raw_raw.info['sfreq']
        minimum_t = max((current_events[0][0] / sampling) -0.5, 0)
        maximum_t = min((current_events[-1][0] / sampling) + 1.5, \
                         raw_raw.n_times)
        cropped_raw = raw_raw.crop(tmin=minimum_t, tmax=maximum_t)
        cropped_events = mne.find_events(cropped_raw)
        assert cropped_events.shape[0] == current_events.shape[0]

        ### EEG: Low-pass at 80 Hz
        picks_eeg = mne.pick_types(raw_raw.info, eeg=True, eog=False)
        cropped_raw.filter(l_freq=None, \
                           h_freq=80, \
                           picks=picks_eeg)

        ### EOG: Band-pass 1-50 Hz the EOG channels only to avoid problems with autoreject
        ### (following the reproducible pipeline on Wakeman&Henson)
        picks_eog = mne.pick_types(raw_raw.info, eeg=False, eog=True)
        cropped_raw.filter(l_freq=1., \
                       h_freq=50., \
                       picks=picks_eog)

        # Epoch the data

        epochs = mne.Epochs(raw=raw_raw, \
                            events=current_events, \
                            tmin=-0.1, tmax=1.1, \
                            preload=True)

        ### Reducing to a sample rate of 256
        epochs.decimate(8)

        # Finding bad channels in EEG

        bad_channels_finder = autoreject.AutoReject(picks=picks_eeg, \
                                                    random_state=1, \
                                                    n_jobs=os.cpu_count())
        epochs, autoreject_log = bad_channels_finder.fit_transform(epochs, \
                                                            return_log=True)

        reject = autoreject.get_rejection_threshold(epochs.copy(), \
                                                    ch_types='eeg')
        epochs.drop_bad(reject=reject)


        mne.set_eeg_reference(epochs, \
                              ref_channels='average', \
                              ch_type='eeg')

        ### ICA

        ### Creating a hi-pass filtered version of the raw data for ICA
        hipass_raw = raw_raw.copy().filter(l_freq=1., \
                                           h_freq=None)
        ica = ICA(random_state=1, n_components=0.999)
        ica.fit(hipass_raw)
        del hipass_raw

        ### EOG detection

        eog_epochs = create_eog_epochs(raw_raw, \
                                       tmin=-.5, tmax=.5, \
                                       preload=True)
        eog_epochs.decimate(8)
        eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)

        # removing Individual Components
        print('Removing Indipendent Components')
        ica.exclude = []
        ica.exclude.extend(eog_inds)

        ica.apply(epochs)

        ### Dropping EOG channels
        epochs.drop_channels(eog_channels)
        ### Dropping the stim channel
        epochs.drop_channels(['Status'])

        epochs_list.append(epochs) 

        ### Adding current events dictionary to general events dictionary before dropping to data frame
        all_events_dict = {h : [l[h_i] for l in data] for h_i, h in enumerate(header)}
        for k, v in all_events_dict.items():
            if k not in all_events.keys():
                all_events[k] = v
            else:
                all_events[k].extend(v)

        ### Preparing the list of epochs actually present in the epochs
        for e_i, e in enumerate(epochs.drop_log):
            if len(e) == 0:
                current_epoch = events_from_file[e_i]
                present_events.append([current_epoch[1], \
                                       current_epoch[4], \
                                       current_epoch[6], \
                                       ]) # word, accuracy, awareness

    all_epochs = mne.concatenate_epochs(epochs_list)
    assert len(all_epochs) == len(present_events)
    
    ### Writing epochs to file
    epochs_file = 'sub-{:02}_epo.fif'.format(s)
    all_epochs.save(os.path.join(output_folder, epochs_file), \
                overwrite=True)

    ### Writing events to file
    with open(os.path.join(output_folder, 'sub-{:02}_epo.events'.format(s)), 'w') as o:
        o.write('Word\tAccuracy\tAwareness\n')
        for e in present_events:
            for v in e:
                o.write('{}\t'.format(v))
            o.write('\n')
