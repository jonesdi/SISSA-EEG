import mne
import os
import pandas
import tqdm
import matplotlib
matplotlib.use('Agg')

import numpy as np

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject import get_rejection_threshold

### load the stimulus set and translate them into events
Stimuli = pandas.read_csv('stimuli_final.csv', delimiter=';')
trial_stimuli = [k for k in range(30)] + [k for k in range(40, 50)]
all_events_ids = {Stimuli['word'][stim] : stim+10 for stim in trial_stimuli}

main_folder = '/mnt/c/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing'

for s in tqdm(range(3, 18)):

    epochs_list = list()
    events_list = list()

    eeg_folder = 'subject{}/sub-{:02}_eeg'.format(s, s)
    events_folder = 'subject{}/sub-{:02}_events'.format(s, s)
    output_folder = 'sub-{:02}'.format(s)
    os.system('rm -r {}'.format(os.path.join(main_folder, output_folder)))
    os.makedirs(os.path.join(main_folder, output_folder), exist_ok=True)

    for r in range (1, 33):

        raw_raw = mne.io.read_raw_bdf(os.path.join(main_folder, eeg_folder, 'Testdata{}.bdf'.format(r)), preload=True)
        sampling = raw_raw.info['sfreq']
        events = mne.find_events(raw_raw)
        cropped_raw = raw_raw.copy().crop(tmin=(events[0][0]/sampling)-0.5, tmax=(events[-1][0]/2048)+2.)
        del events
        events = mne.find_events(cropped_raw)
        events_filter = [k[2] for k in events]
        events_ids = {k : v for k, v in all_events_ids.items() if v in events_filter}

        cropped_raw.set_channel_types({'{}{}'.format(letter, number) : 'eeg' for number in range(1, 33) for letter in ['A', 'B', 'C', 'D']})
        cropped_raw.set_channel_types({'EXG1' : 'eog', 'EXG2' : 'eog'})
        #raw.set_channel_types({ch_name : 'misc' for ch_name in ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']})
        cropped_raw.drop_channels(['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'])

        delay = int(round(0.0345 * sampling))
        events[:, 0] = events[:, 0] + delay

        ### Band pass

        ### Low-pass at 100 Hz
        cropped_raw.filter(None, 40, fir_design='firwin')

        ### High-pass EOG at 1Hz to get reasonable thresholds in autoreject
        picks_eog = mne.pick_types(cropped_raw.info, eeg=False, eog=True)
        #cropped_raw.filter(1., None, picks=picks_eog, fir_window='hann', fir_design='firwin')
        hipass_raw = cropped_raw.copy().filter(1., None, picks=picks_eog)

        ### ICA

        print('Now moving to ICA estimation...')

        # n_components depends on the proportion of variance explained (0.98)

        ica = ICA(random_state=1, n_components=0.999)
        #picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
        #ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12), decim=11)
        ica.fit(hipass_raw)

        #rep = mne.Report()
        #fig = ica.plot_sources(cropped_raw)
        #fig = ica.plot_sources(cropped_raw)
        #rep.add_figs_to_section(fig, captions='ica prova')
        #fig_two = mne.pick_channels_regexp(cropped_raw.ch_names, regexp=r'(EXG[0-9])')
        #f_two = cropped_raw.plot(order=fig_two, n_channels=len(fig_two))
        #rep.add_figs_to_section(f_two, captions='plot c prova')
        #rep.save('prova.html', overwrite=True, open_browser=False)
        #with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
            #print('  Fit {} components (explaining at least {}% of the variance)'.format(ica.n_components_))
            #o.write('  Fit {} components (explaining at least {}% of the variance)\n\n'.format(ica.n_components_, 98))
        #ica.save(os.path.join(main_folder, output_folder, 'ica_solution.ica')

        raw = cropped_raw.copy()
        del hipass_raw, cropped_raw

        ### Epoching

        ###############################################################################
        # We define the events and the onset and offset of the epochs

        tmin = -0.2
        tmax = 1.5  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
        reject_tmax = 1.0  # duration we really care about
        random_state = 1

        baseline = (None, 0)

        ### KEY POINT - DO WE WANT TO INTERPOLATE?
        #raw.interpolate_bads()

        ### projection=True applies the reference from average directly
        raw.set_eeg_reference(ref_channels='average', projection=True, ch_type='eeg')

        picks = mne.pick_types(raw.info, eeg=True, stim=True, eog=False, exclude=())

        # Epoch the data
        print('  Epoching')
        epochs = mne.Epochs(raw=raw, events=events, event_id=events_ids, tmin=tmin, tmax=tmax, picks=picks, reject_tmax=reject_tmax)

        ### EOG detection

        eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)
        eog_epochs.decimate(5)
        eog_epochs.load_data()
        eog_epochs.apply_baseline((None, None))
        eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)
        with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
            print('Found {} EOG indices'.format(len(eog_inds)))
            o.write('Found {} EOG indices\n\n'.format(len(eog_inds)))

        # applying ICA
        print('Applying ICA')
        ica.exclude = []
        ica.exclude.extend(eog_inds)

        epochs.load_data()
        ica.apply(epochs)

        print('Getting rejection thresholds')
        reject = get_rejection_threshold(epochs.copy().crop(None, tmax=reject_tmax), random_state=random_state)
        epochs.drop_bad(reject=reject)
        
        epochs_list.append(epochs)

        print('Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))

        with open(os.path.join(main_folder, output_folder, 'sub-{:02}_preprocessing_log.txt'.format(s)), 'a') as o:
            o.write('  Dropped {}% of epochs\n\n'.format(epochs.drop_log_stats()))
            for epoch_index, tup in enumerate(epochs.drop_log):
                o.write('event {}\trejection reason: {}\n'.format(epoch_index, tup))

        ### Merging original events log with their 'good' or 'rejected' status

        epochs_info = ['good' if len(k) == 0 else 'rejected' for i, k in enumerate(epochs.drop_log)]
        assert len([k for k in epochs_info if k == 'good']) == len(epochs)

        events_raw = pandas.read_csv(os.path.join(main_folder, events_folder, 'run_{:02}_events_log.csv'.format(r)))
        events_info = [(events_raw['Word'][index], events_raw['Group'][index], events_raw['Trigger code'][index], events_raw['Prediction outcome'][index], events_raw['Certainty'][index]) for index in range(20)]

        with open(os.path.join(main_folder, output_folder, 'sub-{:02}_events_rejected_or_good.txt'.format(s)), 'a') as o:
            for result_index, result in enumerate(epochs_info):
                for element in events_info[result_index]:
                    o.write('{}\t'.format(element))
                o.write('{}\n'.format(result))

        del raw, ica, epochs, eog_epochs

    subject_epochs = mne.concatenate_epochs(epochs_list)

    print('Writing to file')
    subject_epochs.save(os.path.join(main_folder, output_folder, 'sub-{:02}_highpass-{}Hz-epoched-concatenated.fif'.format(s, 100)))
