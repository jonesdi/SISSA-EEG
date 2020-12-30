import mne
import collections
import numpy

def get_searchlight_groups():

    montage = mne.channels.make_standard_montage('biosemi128', head_size=0.088)
    positions = montage.get_positions()['ch_pos']
    ### Rescaling so that the minimum value is 0
    scaler = list()
    for i in range(3):
        scaler.append(abs(min([v[i] for k, v in positions.items()])))
    positions = {k : numpy.add(v, scaler) for k, v in positions.items()} 

    ### Now obtaining the searchlight group with tolerance 20mm
    searchlight_groups = collections.defaultdict(list)
    for channel, position in positions.items():
        neighbour_tolerance = [(k-0.02, k+0.02) for k in position]
        for other_channel, other_position in positions.items():
            if other_channel != channel:
                marker = [False, False, False]
                for axis_index, window in enumerate(neighbour_tolerance):
                    if other_position[axis_index] >= window[0] and other_position[axis_index] <= window[1]:
                        marker[axis_index] = True
                if False not in marker:
                    searchlight_groups[channel].append(other_channel)

    return searchlight_groups
