import os
import numpy
import itertools
import math

from matplotlib import pyplot
from scipy import stats

from orthographic_measures import coltheart_N, OLD_twenty

### Reading the Glasgow Norms

with open('glasgow_norms.csv') as i:
    lines = [l.strip().split(',') for l in i.readlines()]
header = lines[0]
relevant_variables = ['Words', 'FAM', 'IMAG']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_variables]
values = [[l[i] for i in indices] for l in lines[2:]]
en_fam_imag = {l[0] : (float(l[1]), float(l[2])) for l in values}

### Reading the Italian norms

with open('lexvar.csv') as i:
    lines = [l.strip().split(',') for l in i.readlines()]
header = [l.lstrip() for l in lines[0]]
relevant_variables = ['WORD', 'FAM', 'IMAG']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_variables]
values = [[l[i] for i in indices] for l in lines[2:]]
it_fam_imag = {l[0].lower() : (float(l[1]), float(l[2])) for l in values}

### Reading the frequencies from itwac

with open('itwac_absolute_frequencies_50k.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]
    
### Transforming the frequencies by their logarithm

freqs = {l[1] : math.log(int(l[2])) for l in lines}

### Opening the Kremer & Baroni word list
with open('concept-measures_it.txt', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]

headers = lines[0]

### Excluding non-animal living things
excluded = ['body_part', 'vegetable', 'fruit']
data = [l for l in lines[1:] if l[1] not in excluded \
                                and l[3] in freqs.keys()]

### Selecting the relevant indices: category, word, length, log frequency
relevant_indices = [1,\
                    3, \
                    #5, \
                    6, \
                    ]
headers_mapping = {'ConceptClass' : 'category', \
                   'ConceptLemma' : 'word', \
                   'No.Letters' : 'length', \
                   'logFreqWaCKy' : 'log_frequency', \
                   }

### Selecting the natural categories
animal_cats = ['mammal', 'bird']
### Correcting words to their recorded version in ItWac
vocabulary_mapping = {'pullover' : 'maglione', \
                      'passero' : 'passerotto'}

### Putting all words and variables into a dictionary
head_dict = dict()

for h_i, h in enumerate(headers):
    if h_i in relevant_indices:

        ### Category and word
        if h_i <= 3:
            heading_data = [d[h_i].lower() for d in data]

               
            ### Correcting categories and word mentions
            if h_i == 1:
                heading_data = ['object' if w not in animal_cats else 'animal' \
                                                         for w in heading_data]
            if h_i == 3:
                heading_data = [w if w not in vocabulary_mapping.keys() else \
                                    vocabulary_mapping[w] for w in heading_data]

                ### Computing coltheart's N and OLD20 for the words
                colt = coltheart_N(heading_data)
                colt_data = [colt[w] for w in heading_data]
                old = OLD_twenty(heading_data)
                old_data = [old[w] for w in heading_data]
                ### Getting the frequencies from ItWac
                freq_data = [math.log(freqs[w]) for w in heading_data]
                head_dict['coltheart_N'] = colt_data
                head_dict['OLD20'] = old_data
                head_dict['log_frequency'] = freq_data

                ### Collecting familiarity and imageability ratings for Glasgow norms
                en_fam_imag_data = [d[0].lower() for d in data]
                en_missing_words = [w for w in en_fam_imag_data if w not in en_fam_imag.keys()]
                #print('missing words: {}'.format(missing_words))
                baroni_en_fam_data = numpy.array([en_fam_imag[w][0] if w in en_fam_imag.keys() else numpy.nan \
                                                             for w in en_fam_imag_data])
                baroni_en_imag_data = numpy.array([en_fam_imag[w][1] if w in en_fam_imag.keys() else numpy.nan \
                                                               for w in en_fam_imag_data])

                ### Collecting familiarity and imageability ratings for Italian norms
                it_fam_imag_data = heading_data
                it_missing_words = [w for w in it_fam_imag_data if w not in it_fam_imag.keys()]
                #print('missing words: {}'.format(missing_words))
                baroni_it_fam_data = numpy.array([it_fam_imag[w][0] if w in it_fam_imag.keys() else numpy.nan \
                                                             for w in it_fam_imag_data])
                baroni_it_imag_data = numpy.array([it_fam_imag[w][1] if w in it_fam_imag.keys() else numpy.nan \
                                                               for w in it_fam_imag_data])
            
        ### Length and frequency
        else:
            heading_data = [float(d[h_i]) for d in data]

        h = headers_mapping[h]
        head_dict[h] = heading_data

#head_dict['domain'] = ['Artifacts' if d[1] not in natural else 'Natural' for d in data]

### Now reading the Montefinese et al. 2013 word list
with open('13428_2012_291_MOESM3_ESM.txt', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
headers = lines[0]

excluded = ['plants', 'body_parts']
### Removing words that are:
    # infrequent 
    # absent from ItWac
    # absent from the previous dictionary
data = [l for l in lines[1:] if \
              float(l[8])>0. and \
              l[1] not in excluded and \
              l[2] in freqs.keys() and \
              l[2] not in head_dict['word']]

#relevant_indices = [0, \
relevant_indices = [1, \
                    2, \
                    4, \
                    #8, \
                    #9, \
                    ]

headers_mapping = {'CATEGORY' : 'category', \
                   'CONCEPT (IT)' : 'word', \
                   'Length (IT)' : 'length', \
                   'LN_Word_Fr' : 'log_frequency', \
                   #'Familiarity_Rating' : 'familiarity'},\
                   #'DOMAIN' : 'domain',\
                   }
                   
for h_i, h in enumerate(headers):

    if h_i in relevant_indices:
        
        ### Word and category
        if h_i <= 2:
            heading_data = [d[h_i] for d in data]
            ### Turning categories into animal/object
            if h_i == 1:
                heading_data = ['animal' if w=='animals' else 'object' \
                                                  for w in heading_data]
            elif h_i == 2:
                ### Computing coltheart's N and OLD20 for the words
                colt = coltheart_N(heading_data)
                colt_data = [colt[w] for w in heading_data]
                old = OLD_twenty(heading_data)
                old_data = [old[w] for w in heading_data]
                head_dict['coltheart_N'].extend(colt_data)
                head_dict['OLD20'].extend(old_data)
                ### Getting the frequencies from ItWac
                freq_data = [math.log(freqs[w]) for w in heading_data]
                head_dict['log_frequency'].extend(freq_data)

                ### Collecting familiarity and imageability ratings for Glasgow norms
                en_fam_imag_data = [d[3].lower() for d in data]
                en_missing_words.extend([w for w in en_fam_imag_data if w not in en_fam_imag.keys()])
                en_fam_data = numpy.array([en_fam_imag[w][0] if w in en_fam_imag.keys() else 0 \
                                                             for w in en_fam_imag_data])
                en_imag_data = numpy.array([en_fam_imag[w][1] if w in en_fam_imag.keys() else 0 \
                                                               for w in en_fam_imag_data])
                head_dict['en_familiarity'] = numpy.concatenate((baroni_en_fam_data, \
                                                                            en_fam_data), axis=0)
                head_dict['en_imageability'] = numpy.concatenate((baroni_en_fam_data, \
                                                                           en_imag_data), axis=0)

                ### Collecting familiarity and imageability ratings for Italian norms
                #fam_imag_data = [d[3].lower() for d in data]
                it_fam_imag_data = heading_data
                it_missing_words.extend([w for w in it_fam_imag_data if w not in it_fam_imag.keys()])
                it_fam_data = [it_fam_imag[w][0] if w in it_fam_imag.keys() else 0 \
                                                             for w in it_fam_imag_data]
                it_imag_data = [it_fam_imag[w][1] if w in it_fam_imag.keys() else 0 \
                                                               for w in it_fam_imag_data]
                head_dict['it_familiarity'] = numpy.concatenate((baroni_it_fam_data, \
                                                                             it_fam_data), axis=0)
                head_dict['it_imageability'] = numpy.concatenate((baroni_it_fam_data, \
                                                                            it_imag_data), axis=0)

        ### Length and frequency
        else:
            heading_data = [float(d[h_i]) for d in data]

        h = headers_mapping[h]
        ### Extending the new 
        head_dict[h].extend(heading_data)

assert len(set([len(v) for k, v in head_dict.items()])) == 1

number_words = list(set([len(v) for k, v in head_dict.items()]))[0]

animal_indices = [w_i for w_i, w in enumerate(head_dict['category']) if w=='animal']
animal_averages = {k : numpy.nanmedian([v[i] for i in animal_indices]) for k, v in head_dict.items() if k not in ['word', 'category']}

object_dict = {k : [v[i] for i in range(number_words) if i not in animal_indices] for k, v in head_dict.items()}

object_counter = {w : [0 for i in range(len(animal_averages.keys()))] for w in object_dict['word']}

for score_i, score_data in enumerate(animal_averages.items()):

    score = score_data[0]
    score_avg = score_data[1]

    score_list = list()
    
    for w_i, w in enumerate(object_dict['word']):

        word_value = object_dict[score][w_i]
        diff = (word_value-score_avg)**2
        score_list.append(diff)
    
    sorted_indices = [k for k, v in sorted(enumerate(score_list), key=lambda item : item[1], reverse=True)]
    for s_i, s in enumerate(sorted_indices):
        word = object_dict['word'][s]
        if score == 'length':
            s_i += 10
        object_counter[word][score_i] += s_i

final_scores = [w for w, s in sorted([(k, sum(v)) for k, v in object_counter.items()], key=lambda item : item[1], reverse=True)]
chosen_words = final_scores[:16]
object_indices = [w_i for w_i, w in enumerate(head_dict['word']) if w in chosen_words]

### Writing the list to file

with open('chosen_words.txt', 'w') as o: 
    o.write('Word\tCategory\n')

    for index in animal_indices:
        animal = head_dict['word'][index]
        if animal == 'passerotto':
            animal = passero
        o.write('{}\tanimal\n'.format(animal))
        
    for index in object_indices:
        current_object = head_dict['word'][index]
        o.write('{}\tobject\n'.format(current_object))

### Plotting
for k in animal_averages.keys():

    animal_data = [head_dict[k][i] for i in animal_indices]
    object_data = [head_dict[k][i] for i in object_indices]
    data = [[k for k in animal_data if k>0], [k for k in object_data if k>0]]
    labels = ['animals', 'objects']

    fig, ax = pyplot.subplots(figsize=(12,5))
    ax.boxplot(data, labels=labels)
    ax.set_title('Measures of {} for the selected words'.format(k))
    os.makedirs('plots', exist_ok=True)
    pyplot.savefig('plots/{}.png'.format(k))
