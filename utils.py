import collections
import numpy
import random

from psychopy import visual, event, core

def format_instr(win, text):
    output = visual.TextStim(win, text, color=[.8,.8,.8], pos=[0,0], ori=0, wrapWidth=700)
    return output

def print_instr(win, instr_type, wait):
    instr_type.draw(win=win)
    win.flip()
    event.waitKeys(keyList=['space'])
    win.flip()
    core.wait(wait)

def draw(win, stimulus, cycles):
    for _ in range(cycles):
        stimulus.draw(win=win)
        win.flip()

def randomize_splits(indices, runs): # creates a dictionary for either 16 or 32 runs

    runs_dict = collections.defaultdict(list)
    counter  = {k : 0 for k in indices}

    for run in range(runs):

        stimuli_indices = [n for n in numpy.random.choice([k for k in counter.keys()], size=10, replace=False)] # randomizes the indices
        run_counter = 0

        for stimulus_index in stimuli_indices:
            if stimulus_index not in runs_dict[run] and counter[stimulus_index] < runs/2 and run_counter <5: # some conditions to make sure stimuli are not repeated
                if counter[stimulus_index] > min([v for k, v in counter.items()]): # makes sure all stimuli are actually balanced in frequency
                    stimuli_indices.append(stimulus_index)
                else:
                    runs_dict[run].append(stimulus_index)
                    counter[stimulus_index] += 1
                    run_counter += 1

        # Check list for repeated items within a run
        repeat_counter = collections.defaultdict(int)
        for i in runs_dict[run]:
            repeat_counter[i] += 1
        assert max([v for k, v in repeat_counter.items()]) == 1

    # Check dict for missing/doubled items within a run
    item_counter = collections.defaultdict(int)
    for r, indices in runs_dict.items():
        for index in indices:
            item_counter[index] += 1
    assert max([v for k, v in item_counter.items()]) == 16

    return runs_dict

def create_run_splits(runs=32): # creates a dictionary for either 16 or 32 runs

    animal_targets = randomize_splits([k for k in range(10)], runs)
    object_targets = randomize_splits([k for k in range(10, 20)], runs)
    animal_fillers = randomize_splits([k for k in range(20, 30)], runs)
    object_fillers = randomize_splits([k for k in range(40, 50)], runs)

    final_runs = collections.defaultdict(list)
    for k in range(runs):
        final_runs[k] += animal_targets[k]
        final_runs[k] += object_targets[k]
        final_runs[k] += animal_fillers[k]
        final_runs[k] += object_fillers[k]

    final_runs = {k : random.sample(v, k=len(v)) for k, v in final_runs.items()}

    return final_runs
