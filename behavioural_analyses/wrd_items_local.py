import os
<<<<<<< HEAD
import numpy
import pandas
import collections

import statsmodels.api as sm
import statsmodels.formula.api as smf

from orthographic_measures import coltheart_N, OLD_twenty

def mixed_model(dependent_variable, fixed, random, data_frame):
    md = smf.mixedlm('{} ~ {}'.format(dependent_variable, fixed), data_frame, re_formula=random, groups=data_frame['subject'])
    mdf = md.fit(method=["lbfgs"])
    print(mdf.summary())
=======
>>>>>>> 2e03cec974770c97b0cbf0aea1db0b94848ecc70

### Reading the stimuli file
with open('../lab_experiment/stimuli_final.csv') as i:
    stimuli = [l.strip().split(';')[:3] for l in i.readlines()][1:]

<<<<<<< HEAD
word_to_cat = {w[0] : w[1] for w in stimuli}

### Computing the orthographic neighborhood measures

colt = coltheart_N([w[0] for w in stimuli])

OLD = OLD_twenty([w[0] for w in stimuli])

### Reading the frequencies
with open('/import/cogsci/andrea/dataset/co_occurrences/itwac/itwac_absolute_frequencies_50k.txt') as i:
    frequencies = [l.strip().split('\t') for l in i.readlines()][1:]

frequencies = {l[1] : float(l[2]) for l in frequencies}

for k in stimuli:
    assert k[0] in [k for k in frequencies.keys()]

### Creating the frequencies dictionary
frequencies = {k[0] : frequencies[k[0]] for k in stimuli}
mean_freq = numpy.average([v for k, v in frequencies.items()])
std_freq = numpy.std([v for k, v in frequencies.items()])
z_frequencies = {k : (v-mean_freq)/std_freq for k, v in frequencies.items()}

### Creating the lengths dictionary
lengths = {k[0] : len(k[0]) for k in stimuli}
mean_len = numpy.average([v for k, v in lengths.items()])
std_len = numpy.std([v for k, v in lengths.items()])
z_lengths = {k : (v-mean_len)/std_len for k, v in lengths.items()}

### Path to the original files
#data_folder = 'C:/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing/raw_files'
data_folder = '/import/cogsci/andrea//dataset/neuroscience/conscious_unconscious_processing/behavioural_events_log/'

### Creating the dict container for the data
data_dict = {'subject' : list(), \
             'word' : list(), \
             'frequency' : list(), \
             'length' : list(), \
             'coltheart_N' : list(), \
             'OLD_20' : list(), \
             'target' : list(), \
             'trial' : list(), \
             'category' : list(), \
             'accuracy' : list(), \
             'awareness' : list(), \
            }

### Reading the participants' logs
for s in range(2, 18):
    word_counter = collections.defaultdict(int)

    current_csv_path = os.path.join(data_folder, 'sub-{:02}_events'.format(s))
    sub_list = list()
    for r in range(1,33):
        ### Correcting for naming mistake
        if s == 2:
            r = r - 1
        with open(os.path.join(current_csv_path, 'run_{:02}_events_log.csv'.format(r))) as i:
            current_lines = [l.strip().split(',')[1:] for l in i.readlines()]
        data_info = current_lines[0][:-1]
        data = [[w for w in l][:-1] for l in current_lines[1:]]

        for d in data:
            word = d[0]
            word_counter[word] += 1
            target = -.5 if d[1] == 'filler' else .5
            category = -.5 if word_to_cat[word] == 'object' else .5

            accuracy = 0 if d[3] == 'wrong' else 1
            data_dict['subject'].append(s)
            data_dict['word'].append(word)
            data_dict['frequency'].append(z_frequencies[word])
            data_dict['length'].append(z_lengths[word])
            data_dict['coltheart_N'].append(colt[word])
            data_dict['OLD_20'].append(OLD[word])
            data_dict['target'].append(target) 
            data_dict['category'].append(category)
            data_dict['trial'].append(word_counter[word])
            data_dict['accuracy'].append(accuracy-.5)
            data_dict['awareness'].append(float(d[5])-2.)

data_frame = pandas.DataFrame.from_dict(data_dict)
data_frame.to_csv('behav_data_long_format.csv')

data_frame = pandas.read_csv('behav_data_long_format.csv')

mixed_model(dependent_variable, fixed, random, data_frame)
=======
### Reading lexvar
with open('lexvar.csv') as i:
    lexvar = [l.strip().split(',') for l in i.readlines()]

heading = list()
for index, metric_measure in enumerate(zip(lexvar[0], lexvar[1])):
    metric = metric_measure[0]
    measure = metric_measure[1]
    if metric == '':
        for i in range(index, -1, -1):
            if lexvar[0][i] != '':
                metric = lexvar[0][i]
                break
    value = '{}_{}'.format(metric, measure)
    heading.append(value)

    
chosen_variables = ['WORD_Italian', '   FAM_mean', 'IMAG_mean', 'CONC_mean','Adult WrtFQ_ILC', 'Adult WrtFQ_CoLFIS', 'Adult_NSIZE', 'Adult_BIGR', 'Adult_SYL', 'Lexical_LET']
relevant_indices = [i[0] for i in enumerate(heading) if i[1] in chosen_variables]

lexvar = [[l[i] for i in relevant_indices] for l in lexvar if l[0].lower() in [w[0] for w in stimuli]]
import pdb; pdb.set_trace()
>>>>>>> 2e03cec974770c97b0cbf0aea1db0b94848ecc70
