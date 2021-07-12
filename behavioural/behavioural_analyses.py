import matplotlib
import numpy
import os

from scipy import stats

### Reading triggers
import sys
additional_path = 'lab/lab_two'
sys.path.append(additional_path)

from utils_two import read_words_and_triggers

word_to_trigger, questions = read_words_and_triggers(\
                               additional_path=additional_path, \
                                         return_questions=True)

def read_results(args, exp):

    relevant_indices = [0, 2, 3, 4, 5] if args.experiment_id == 'two' \
                       else [1,4,6]

    container_dict = dict()
   
    #for s in range(1, exp.n_subjects+1):
    for s in range(1, 6+1):

        sub_folder = os.path.join(args.data_folder, \
                         'sub-{:02}'.format(s), \
                         'sub-{:02}_events'.format(s))
        assert os.path.exists(sub_folder)
        files = os.listdir(sub_folder)

        for f in files:
            with open(os.path.join(sub_folder, f)) as i_f:
                lines = [l.strip().split('\t') for l in i_f.readlines()]
            header = lines[0]
            data = lines[1:]
            ### Initializing the dictionary
            for i in relevant_indices:
                if header[i] not in container_dict.keys():
                    container_dict[header[i]] = list()
            if 'subject' not in container_dict.keys():
                container_dict['subject'] = list()
            
            ### Fix for experiment 2
            for l in data:
                assert len(l) == len(header)
                #if len(l) < len(header):
                    #l.insert(0, '_')
            ### Adding the data
            for l in data:
                for i in relevant_indices:
                    if i in [2, 6]:
                        value = int(l[i])
                    else:
                        '''
                        ### Correction not needed anymore
                        if i == 4 and args.experiment_id == 'two':
                            w = l[0]
                            question = l[1]
                            if w!='_':
                                if question in questions[w]:
                                    if l[4] == 'correct':
                                        l[4] = 'wrong'
                                    elif l[4] == 'wrong':
                                        l[4] = 'correct'
                                    else:
                                        raise RuntimeError('There is a problem with reading'\
                                                           ' the original files')
                        '''
                        value = l[i]
                    container_dict[header[i]].append(value)
                container_dict['subject'].append(s)

            '''
            ### Writing to file the corrected versions
            
            file_folder = sub_folder.replace('raw', 'corrected_events')
            os.makedirs(file_folder, exist_ok=True)
            if int(s) == 1:
                for err_run, right_run in {1 : 17, 2 : 18, 4 : 19, 5: 20, 6: 21, 7 : 22, 8: 23, 9: 24}.items():
                    f = f.replace('sub-02_run-{:02}'.format(err_run), 'sub-01_run-{:02}'.format(right_run))
            file_path = os.path.join(file_folder, f)
            
            with open(file_path,  'w', encoding='utf-8') as o:
                for h in header:
                    o.write('{}\t'.format(h))
                o.write('\n')
                for l in data:
                    for value in l:
                        o.write('{}\t'.format(value))
                    o.write('\n')
            '''

    file_path = os.path.join('results', 'behavioural', args.experiment_id)
    os.makedirs(file_path, exist_ok=True)
    file_name = os.path.join(file_path, 'experiment_{}.csv'.format(\
                                     args.experiment_id))    

    with open(file_name, 'w') as o:
        for k in container_dict.keys():
            o.write('{},'.format(k))
        o.write('\n')
        for i in range(len(container_dict[list(container_dict.keys())[0]])):
            for k in container_dict.keys():
                value = container_dict[k][i]
                o.write('{},'.format(value))
            o.write('\n')

    return container_dict
