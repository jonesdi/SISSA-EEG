import os
import gensim
import scipy
import itertools
import tqdm
import argparse
import nltk

from tqdm import tqdm
from nltk.corpus import wordnet

parser = argparse.ArgumentParser()
parser.add_argument('--computational_model', choices=['w2v', 'wordnet'], required=True, help='Indicates for which model to extract the similarities')
args = parser.parse_args()

### Housekeeping

output_folder = 'computational_models/{}'.format(args.computational_model)
os.makedirs(output_folder, exist_ok=True)

### Reading words

words = []

if args.computational_model != 'wordnet':

    with open('../lab_experiment/stimuli_final.csv', 'r') as f:
        for index, l in enumerate(f):
            if index != 0:
                l = str(l).strip().split(';')
                #if l[2] == 'target':
                    #words.append(l[0])
                words.append(l[0])

    word_combs = [k for k in itertools.combinations(words, r=2)]

    for i in [100000, 150000]:
        with open(os.path.join(output_folder, 'sims_w2v_vocab_{}'.format(i)), 'w') as o:
            w2v = gensim.models.Word2Vec.load('../../dataset/corpora/itwac/w2v_vocab_{}/w2v_itwac_vocab_{}'.format(i, i))
            for word_one, word_two in word_combs:
                current_sim = w2v.similarity(word_one, word_two)
                current_corr = scipy.stats.pearsonr(w2v.wv[word_one], w2v.wv[word_two])[0]
                o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))

 

elif args.computational_model == 'wordnet':

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

    synsets = {w : wordnet.synset(w) for w in words}
    word_combs = [k for k in itertools.combinations(words, r=2)]

    with open(os.path.join(output_folder, 'wordnet.sims'), 'w') as o:
        for w in word_combs:
            sim = wordnet.path_similarity(synsets[w[0]], synsets[w[1]])
            o.write('{}\t{}\t{}\n'.format(eng_to_it[w[0]], eng_to_it[w[1]], sim))

