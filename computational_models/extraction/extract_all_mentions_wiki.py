import os
from tqdm import tqdm

### Reading the list of words

with open('lab/lab_two/chosen_words.txt') as i:
    words = [l.strip().split('\t')[0] for l in i.readlines()][1:]

wiki_files_folder = '/import/cogsci/andrea/dataset/corpora/wikipedia_italian/annotated_it_wiki_article_by_article'

word_mapper = {w : w for w in words}

all_files = list([os.path.join(root, f) for root, direc, files in os.walk(wiki_files_folder) for f in files])

output_folder = os.path.join('it_wiki_sentences_per_word')
os.makedirs(output_folder, exist_ok=True)

### Not acquiring sentences if file already existing
required_words = list()
for w in words:

    file_path = os.path.join(output_folder, '{}.sentences'.format(w))
    if not os.path.exists(file_path):
        required_words.append(w)

sentences_container = {w : list() for w in required_words}

for f in tqdm(all_files):

    with open(f) as i:
        lines = [l.strip().split('\t')[:2] for l in i.readlines()][1:]
    lines = [l for l in lines if 'WORD' not in l]
    
    lines =[l[0] if l[1] not in words else l[1] \
                                for l in lines]
    lines = ' '.join(lines)
    lines = lines.split(' BREAK ')
    for l in lines:
        split_l = l.split()
        for w in required_words:
            if w in split_l:
                l = l.replace('BREAK', '')
                sentences_container[w].append(l)

for w, sents in sentences_container.items():
    file_path = os.path.join(output_folder, '{}.sentences'.format(w))
    with open(file_path, 'w') as o:
        for sent in sents:
            o.write('{}\n'.format(sent))
            
#lines = ['{}.'.format(l.lower()) for l in lines if w in l]
