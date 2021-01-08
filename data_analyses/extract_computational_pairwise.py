import os
import gensim
import scipy
import itertools
import tqdm

from tqdm import tqdm

### Housekeeping

output_folder = 'computational_models'
os.makedirs(output_folder, exist_ok=True)

### Reading words

words = []

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
