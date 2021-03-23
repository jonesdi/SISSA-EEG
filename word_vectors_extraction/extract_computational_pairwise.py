import os
import gensim
import scipy
import itertools
import tqdm
import argparse
import nltk
import numpy
import math
import re

from tqdm import tqdm
from nltk.corpus import wordnet


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])
parser = argparse.ArgumentParser()
parser.add_argument('--computational_model', choices=['simlex-999', 'w2v', 'wordnet', 'orthography'], required=True, help='Indicates for which model to extract the similarities')
args = parser.parse_args()

### Housekeeping

output_folder = '../rsa_analyses/computational_models/{}'.format(args.computational_model)
os.makedirs(output_folder, exist_ok=True)

### Reading words

words = []

#if args.computational_model != 'wordnet':
if 1 != 1:

    with open('../lab_experiment/stimuli_final.csv', 'r') as f:
        for index, l in enumerate(f):
            if index != 0:
                l = str(l).strip().split(';')
                #if l[2] == 'target':
                    #words.append(l[0])
                words.append(l[0])

    word_combs = [k for k in itertools.combinations(words, r=2)]

    if args.computational_model == 'w2v':
        for i in [100000, 150000]:
            with open(os.path.join(output_folder, 'sims_w2v_vocab_{}'.format(i)), 'w') as o:
                w2v = gensim.models.Word2Vec.load('../../dataset/corpora/itwac/w2v_vocab_{}/w2v_itwac_vocab_{}'.format(i, i))
                for word_one, word_two in word_combs:
                    current_sim = w2v.similarity(word_one, word_two)
                    current_corr = scipy.stats.pearsonr(w2v.wv[word_one], w2v.wv[word_two])[0]
                    o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))

    elif args.computational_model == 'orthography':
        sims = dict()
        for word_one, word_two in word_combs:
            current_sim = levenshtein(word_one, word_two)
            sims[(word_one, word_two)] = -current_sim

        #l2_norm = math.sqrt(sum([v*v for k, v in sims.items()]))
        #l1_norm = sum([-v for k, v in sims.items()])
        std = numpy.nanstd([v for k, v in sims.items()])
        mean = numpy.nanmean([v for k, v in sims.items()])
        normalized_sims = {k : ((float(v)-mean)/std) for k, v in sims.items()}
        
        with open(os.path.join(output_folder, 'orthography.sims'), 'w') as o:
            for word_tuple, sim in normalized_sims.items():
                o.write('{}\t{}\t{}\n'.format(word_tuple[0], word_tuple[1], sim))

#elif args.computational_model == 'wordnet' or args.computational_model == 'simlex-999':
elif args.computational_model == 'simlex-999':

    eng_to_it = dict()

    with open('../lab_experiment/stimuli_final.csv', 'r') as stimuli_file:
        for i, l in enumerate(stimuli_file):
            if i > 0: 
                l = l.strip().split(';')
                w_and_n = l[3]
                if '_' in w_and_n:
                    w_and_n = w_and_n.split('_')
                    word = '{}.n.{}'.format(w_and_n[0].replace(' ', '_'), w_and_n[1])
                else:
                    word = '{}.n.01'.format(w_and_n.replace(' ', '_'))
                words.append(word)
                eng_to_it[word] = l[0] 

    word_combs = [k for k in itertools.combinations(words, r=2)]

    if args.computational_model == 'simlex-999':

        simlex_dict = dict()
        with open('../../../casa_nuova/lib/python3.8/site-packages/gensim/test/test_data/simlex999.txt') as i:
            simlex999 = [l.strip().split('\t') for l in i.readlines()][2:]
        for l in simlex999:
            simlex_dict[(l[0], l[1])] = float(l[2])
            simlex_dict[(l[1], l[0])] = float(l[2])

        counter = list()
        output_dict = dict()
        for c in word_combs:
            if isinstance(c, tuple):
                c = tuple([re.sub('\...+', '', w) for w in c])
                if isinstance(c, tuple):
                   
                    if c not in simlex_dict.keys():
                        counter.append(c)
                    else:
                        try:
                            output_dict[c] = simlex_dict[c]
                        except IndexError:
                            output_dict[c] = simlex_dict[(c[1], c[0])]

        #assert len(counter) == 0
 
        import pdb; pdb.set_trace()

    synsets = {w : wordnet.synset(w) for w in words}

    with open(os.path.join(output_folder, 'wordnet.sims'), 'w') as o:
        for w in word_combs:
            sim = wordnet.path_similarity(synsets[w[0]], synsets[w[1]])
            o.write('{}\t{}\t{}\n'.format(eng_to_it[w[0]], eng_to_it[w[1]], sim))

