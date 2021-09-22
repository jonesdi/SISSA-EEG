import argparse
import autoreject
import matplotlib
import mne
import multiprocessing
import numpy
import os
import pandas
import tqdm

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject.utils import interpolate_bads

### Reading triggers
import sys
additional_path = '../lab/lab_two'
sys.path.append(additional_path)

from utils_two import read_words_and_triggers

def preprocess_eeg(s):

    epochs_list = list()
    present_events = list()
    '''
    ### Useless
    all_events = dict()
    '''

    word_to_trigger = read_words_and_triggers(additional_path=additional_path)

    ### Preparing the folders
    subject_folder = 'sub-{:02}'.format(s)
    eeg_folder = os.path.join(args.data_folder, subject_folder, 'sub-{:02}_eeg'.format(s))
    events_folder = os.path.join(args.data_folder, subject_folder, 'sub-{:02}_events'.format(s))

    output_folder = os.path.join(args.data_folder.replace('raw', 'preprocessed'), subject_folder)
    os.makedirs(os.path.join(output_folder), exist_ok=True)

    ### Preparing some useful variables
    if args.experiment_id == 'one':

        n_runs = 33
        n_trials = 20
        word_index = 1
        pas_index = 6
        accuracy_index = 4

    elif args.experiment_id == 'two':

        n_runs = 24
        n_trials = 33
        word_index = 0
        pas_index = 2
        accuracy_index = 4

    runs = list(range(1, n_runs))
    ### Fix for subject 14 in exp two
    if args.experiment_id == 'two' and s == 14:
        runs = [r for r in runs if r != 3]

    ### Preprocessing run by run
    for r in runs:

        ### Loading run raw data
        run_path = os.path.join(eeg_folder, 'sub-{:02}_run-{:02}.bdf'.format(s, r))
        try:
            assert os.path.exists(run_path)

        ### Fixing for experiment two
        except AssertionError:
            if s == 10:
                run_path = run_path.replace('sub-10_eeg/', 'sub_10-eeg/').replace('run-', 'eeg-')
                os.path.exists(run_path)

        raw_raw = mne.io.read_raw_bdf(run_path, preload=True, \
                                      eog=eog_channels, \
                                      exclude=excluded_channels, \
                                      verbose=False)
        raw_raw.set_montage(montage)

        ### Extracting events from file
        if args.experiment_id == 'one':
            events_alias = 'run-{:02}.events'.format(r)
        elif args.experiment_id == 'two':
            events_alias = 'sub-{:02}_run-{:02}.events'.format(s, r)
        events_path = os.path.join(events_folder, events_alias)
        with open(events_path) as events_file:
            lines = [l.strip().split('\t') for l in events_file]
        header = lines[0]
        data = lines[1:]
        if args.experiment_id == 'two':
            for l in data:
                if len(l) < len(header):
                    l.insert(0, '')
                assert len(l) == len(header)
                if l[0] == '_':
                    l[0] = ''
                l.append(word_to_trigger[l[0]])
            
        '''
        ### Useless
        all_events_dict = {h : [l[h_i] for l in data] for h_i, h in enumerate(header)}
        all_events_dict['trigger'] = list()
        for w in all_events_dict['Current word']:
            all_events_dict['trigger'].append(word_to_trigger[w])
        assert len(list(set([len(v) for k, v in all_events_dict.items()]))) == 1
        '''

        current_events = mne.find_events(raw_raw)
        ### Correcting for spurious events
        current_events = current_events[current_events[:, 2]<100]
        ### Making sure we do not use missing runs:

        if current_events.shape[0] == 0:
            pass
        else:
            ## Check again the next line
            real_n_trials = n_trials if (s!=14 and args.experiment_id=='two') else n_trials*2
            if current_events.shape[0] < real_n_trials:
                print('sub {} r {} missing some trials'.format(s, r))
            assert current_events.shape[0] <= real_n_trials
            ### Correcting for a mistake in data acquisition
            if s == 1 and args.experiment_id == 'one':
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
                    if args.experiment_id == 'one' and s == 1:
                        trigger = int(e[3])-10
                    else:
                        trigger = int(e[-1])
                    if trigger == current_events[index, 2]:
                        events_from_file.append(e)
                    else:
                        to_be_subtracted += 1

            assert len(events_from_file) == current_events.shape[0]
            for trig_i in range(len(events_from_file)):
                assert events_from_file[trig_i][-1] == current_events[trig_i, -1]
       
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
                                tmin=-0.1, tmax=1., \
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

            '''
            ### Useless
            ### Adding current events dictionary to general events dictionary before dropping to data frame
            for k, v in all_events_dict.items():
                if k not in all_events.keys():
                    all_events[k] = v
                else:
                    all_events[k].extend(v)
            '''

            ### Preparing the list of epochs actually present in the epochs
            for e_i, e in enumerate(epochs.drop_log):
                if len(e) == 0:
                    current_epoch = events_from_file[e_i]
                    assert epochs.events[e_i][-1] == word_to_trigger[current_epoch[0]]
                    present_events.append([current_epoch[word_index], \
                                           current_epoch[accuracy_index], \
                                           current_epoch[pas_index], \
                                           ]) # word, accuracy, awareness

    all_epochs = mne.concatenate_epochs(epochs_list)
    assert len(all_epochs) == len(present_events)
    for ev, erp in zip(present_events, all_epochs.events):
        assert word_to_trigger[ev[0]] == erp[-1]
    
    '''
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
    '''

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, \
                    required=True, help='Folder where \
                    the original data is stored')
parser.add_argument('--experiment_id', choices=['one', 'two'],\
                    required=True, help='Which experiment?')
args = parser.parse_args()

### Channel naming

eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

montage = mne.channels.make_standard_montage(kind='biosemi128')
random_state = 1

number_of_subjects = len(os.listdir(args.data_folder))

targets = list(range(1, number_of_subjects+1))
#targets = list(range(15, 15+1))

'''
with multiprocessing.Pool() as p:
    p.map(preprocess_eeg, targets)
    p.terminate()
    p.join()
'''
for target in targets:
    preprocess_eeg(target)
