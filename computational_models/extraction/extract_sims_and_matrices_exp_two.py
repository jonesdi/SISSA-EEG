import argparse
import gensim
import itertools
import math
import numpy
import os
import random
import re
import scipy
import sys

from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel

from extraction_utils import read_words

sys.path.append('/import/cogsci/andrea/github/')

from grano.plot_utils import confusion_matrix_simple

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')
args = parser.parse_args()

plot_path = os.path.join('..', 'confusion_matrices', args.experiment_id)
os.makedirs(plot_path, exist_ok=True)

it_words, en_words = read_words(args)

assert len(it_words) in [32, 40]
#word_combs = [k for k in itertools.combinations(it_words, r=2)]

output_folder = os.path.join('..', 'similarities', args.experiment_id)
os.makedirs(output_folder, exist_ok=True)

wiki_folder = os.path.join('..', '..', 'resources', 'it_wiki_pages')

### Loooking at Word2Vec similarities

w2v_sims = list()

with open(os.path.join(output_folder, 'w2v.sims'), 'w') as o:
    w2v_file = os.path.join('..', '..', \
                            'resources', \
                            'w2v_it_combined_corpora', \
                            #'w2v_itwac_vocab_150000', \
                            'w2v_it_combined_corpora_vocab_150000'
                            #'w2v_itwac_vocab_150000'\
                            )
    w2v = gensim.models.Word2Vec.load(w2v_file)
    #for word_one, word_two in word_combs:
    for word_one in it_words:
        word_one_list = list()
        for word_two in it_words:
            if word_one != word_two:
                current_sim = w2v.wv.similarity(word_one, word_two)
                current_corr = scipy.stats.pearsonr(w2v.wv[word_one], w2v.wv[word_two])[0]
            else:
                current_sim = 0.
                current_corr = 0.
            word_one_list.append(current_sim)
            o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))
        w2v_sims.append(word_one_list)

### Plotting the matrix

title = 'Confusion matrix for Word2Vec'

file_path = os.path.join(plot_path, 'w2v_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(w2v_sims), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(w2v_sims), vmax=numpy.amax(w2v_sims))

### Collecting co-occurrences and PPMIs

log_coocs = list()
ppmis = list()

'''
with open(os.path.join('..', '..', '..', 'resources', \
                       'itwac_absolute_frequencies.txt'),\
                       encoding='utf-8') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
    total_words = int(re.sub(r'[^\d]+', '', lines[0][-1]))
    word_freqs = {l[1] : float(l[3]) for l in lines[1:]}
'''

cooc_folder = os.path.join('..', '..', 'resources', \
                           'cooc_combined_corpora_window_5')
                           #'co_occurrences_itwac_vocab_150000_window_5_subsampling_False')
for word_one in it_words:
    #relative_freq_one = word_freqs[word_one]

    word_one_path = os.path.join(cooc_folder, word_one[:3], '{}.coocs'.format(word_one))
    assert os.path.exists(word_one_path)
    with open(word_one_path, encoding='utf-8') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]

    word_coocs = {l[0] : math.log(float(l[1]), 2) for l in lines}
    word_ppmis = {l[0] : float(l[3]) for l in lines}

    for w in it_words:
        if w not in word_coocs.keys():
            word_coocs[w] = 0.
            word_ppmis[w] = 0.
 
    word_one_coocs = [word_coocs[w] if w!=word_one else 0. for w in it_words]
    word_one_ppmis = [word_ppmis[w] if w!=word_one else 0. for w in it_words]

    log_coocs.append(word_one_coocs)
    ppmis.append(word_one_ppmis)

with open(os.path.join(output_folder, 'log_cooc.sims'), 'w') as o:
    for w_i_one, word_one in enumerate(it_words):
        for w_i_two, word_two in enumerate(it_words):
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, log_coocs[w_i_one][w_i_two]))

with open(os.path.join(output_folder, 'ppmi.sims'), 'w') as o:
    for w_i_one, word_one in enumerate(it_words):
        for w_i_two, word_two in enumerate(it_words):
            ppmi_score = ppmis[w_i_one][w_i_two]
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, ppmi_score))
'''
### Plotting the matrix for co-ocs

title = 'Confusion matrix for ItWac co-occurrences'

file_path = os.path.join(plot_path, 'coocs_confusion_exp_two.png')
confusion_matrix_simple(numpy.array(coocs), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(coocs), vmax=numpy.amax(coocs))
'''
### Plotting the matrix for the log co-occurrences

title = 'Confusion matrix for log co-occurrence'

file_path = os.path.join(plot_path, 'log_cooc_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(log_coocs, it_words, title, file_path, text=False, \
                        vmin=numpy.amin(log_coocs), vmax=numpy.amax(log_coocs))

### Plotting the matrix for PPMI

title = 'Confusion matrix for PPMI'

file_path = os.path.join(plot_path, 'ppmi_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(ppmis), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(ppmis), vmax=numpy.amax(ppmis))
### Looking at BERT similarities

#model_name = "dbmdz/bert-base-italian-xxl-cased"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")

bert_vectors = dict()

for w in tqdm(it_words):
    
    w_vectors = list()
    wiki_file = os.path.join(wiki_folder, '{}.wiki'.format(w))
    with open(wiki_file, encoding='utf-8') as i:
        final_lines = [l.strip() for l in i.readlines()]
    
    final_lines = final_lines[:6]


    for l in final_lines:

        ### Encoding the sentence
        encoded_input = tokenizer(l, return_tensors='pt')
        
        ### Getting the model output
        output = model(**encoded_input, output_hidden_states=True, \
                       output_attentions=False, return_dict=True)
    
        ### Averaging all layers and all sentences

        mention = list()
        for layer in range(1, 13):
            layer_vec = output['hidden_states'][layer][0, 1:-1, :].detach().numpy()
            mention.append(numpy.average(layer_vec, axis=0))
        mention = numpy.average(mention, axis=0)
        assert mention.shape == (768, )
        w_vectors.append(mention)

    ### Averaging across mentions
    assert len(w_vectors) >= 1
    bert_vectors[w] = numpy.average(w_vectors, axis=0)
     
    '''
    # Encoding individual mentions instead of sentences

    bert_tokens = tokenizer.tokenize(w)
    bert_tokens = bert_tokens if isinstance(bert_tokens, list) else [bert_tokens]
    token_id = tokenizer.convert_tokens_to_ids(bert_tokens)
    if token_id == [100]:
        import pdb; pdb.set_trace()
        ### Finding the relevant indices
        tokenized_input = '_'.join(tokenizer.tokenize(l))
        joined_id = '_'.join(bert_tokens)
        marked_id = re.sub(r'[a-z]+', 'MARK', joined_id)
        tokenized_input = tokenized_input.replace(joined_id, marked_id)
        split_marked = tokenized_input.split('_')
        relevant_indices = [t_i for t_i, t in enumerate(split_marked) if 'MARK' in t]

        encoded_input = tokenizer(l, return_tensors='pt')
        #relevant_indices = [t_i for t_i, t in enumerate(encoded_input['input_ids'][0, :].detach().tolist()) \
                                                           #if t in token_id]
        #if len(relevant_indices) >= 1:
        for rel_i in relevant_indices:

            mention = list()
            ### Extracting layers 3-8
            for layer in range(3, 8):
                try:
                    layer_vec = output['hidden_states'][layer][0, rel_i, :].detach().numpy()
                except IndexError:
                    import pdb; pdb.set_trace()
                mention.append(layer_vec)
            ### Averaging across layers
            mention = numpy.average(mention, axis=0)
            assert mention.shape == (768, )
            w_vectors.append(mention)
    '''
    
### Loooking at BERT similarities

berts = list()

with open(os.path.join(output_folder, 'bert.sims'), 'w') as o:
    for word_one in it_words:
        word_one_list = list()
        for word_two in it_words:
            current_corr = scipy.stats.pearsonr(bert_vectors[word_one], bert_vectors[word_two])[0]
            word_one_list.append(current_corr)
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, current_corr))
        berts.append(word_one_list)

### Plotting the matrix

title = 'Confusion matrix for BERT'

file_path = os.path.join(plot_path, 'bert_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(berts), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(berts), vmax=numpy.amax(berts))

### Collecting Wordnet similarities

### Using the list of it_words in English

en_mapping = {'pork' : 'pig', \
              'cow' : 'cattle', \
              'glass' : 'drinking_glass', \
              'plate' : 'dish'}

en_words = [w if w not in en_mapping.keys() else en_mapping[w] for w in en_words]

if args.experiment_id == 'one':
    synsets = [wordnet.synset(w) for w in en_words]
elif args.experiment_id == 'two':
    synsets = [wordnet.synsets(w)[0] for w in en_words]
     
wordnets = list()

with open(os.path.join(output_folder, 'wordnet.sims'), 'w') as o:
    for w_one_i, w_syns_one in enumerate(synsets):
        w_sims = list()
        for w_two_i, w_syns_two in enumerate(synsets):
            if w_one_i != w_two_i:
                sim = wordnet.path_similarity(w_syns_one, w_syns_two)
            else:
                sim = 0.
            w_sims.append(sim)
            o.write('{}\t{}\t{}\n'.format(it_words[w_one_i], it_words[w_two_i], sim))
        wordnets.append(w_sims)

wordnets = numpy.array(wordnets)

### Plotting the matrix for WordNet

title = 'Confusion matrix for WordNet'

file_path = os.path.join(plot_path, 'wordnet_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(wordnets, it_words, title, file_path, text=False, \
                        vmin=numpy.amin(wordnets), vmax=numpy.amax(wordnets))


### Plotting the models correlations

models = [log_coocs, ppmis, w2v_sims, berts, wordnets]
#models = [w2v_sims, berts, wordnets]
names = ['log co-occurrence', 'PPMI', 'Word2Vec', 'mBERT', 'WordNet']
#names = ['Word2Vec', 'BERT', 'WordNet']

flattened = list()
for m_i, m in enumerate(models):
    new_m = numpy.array([[w[i] for i in range(len(it_words)) if i!=w_i] for w_i, w in enumerate(m)]).flatten()
    flattened.append(new_m)

models_corr = list()
for m_one in flattened:
    m_list = list()
    for m_two in flattened:
        corr = scipy.stats.pearsonr(m_one, m_two)[0]
        m_list.append(corr)
    models_corr.append(m_list)

models_corr = numpy.array(models_corr)
title = 'Confusion matrix comparing the models'

file_path = os.path.join(plot_path, 'all_models_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(models_corr, names, title, file_path, text=True)
