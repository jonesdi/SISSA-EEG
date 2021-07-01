import os

from tqdm import tqdm

### Reading the list of it_words

words_path = os.path.join('..', '..', '..', \
                          'lab', 'lab_two', 'chosen_words.txt') 
with open(words_path, encoding='utf-8') as i:
    all_words = [l.strip().split('\t') for l in i.readlines()][1:]
it_words = [w[0] for w in all_words]

### Creating the output folder
output_folder = os.path.join('..', '..', '..', 'resources', 'it_wiki_pages')
os.makedirs(output_folder, exist_ok=True)

### Converting words into Wikipedia names

wiki_files_folder = '/import/cogsci/andrea/dataset/corpora/wikipedia_italian/annotated_it_wiki_article_by_article'

bert_mapping = {'orso' : 'Ursidae', \
                'cane' : 'Canis_lupus_familiaris', \
                'cavallo' : 'Equus_ferus_caballus', \
                'scimmia' : 'Gorilla', \
                'gabbiano' : 'Larinae', \
                'elefante' : 'Elephantidae', \
                'gatto' : 'Felis_silvestris_catus', \
                'leone' : 'Panthera_leo', \
                'maiale' : 'Sus_scrofa_domesticus', \
                'mucca' : 'Bos_taurus', \
                'serpente' : 'Serpentes', \
                'tigre' : 'Panthera_tigris', \
                'gufo' : 'Asio_otus', \
                'passero' : 'Passer_domesticus', \
                'tenda' : 'Tenda__arredamento_', \
                'piatto' : 'Piatto__stoviglia_', \
                'scopa' : 'Scopa__strumento_', \
                'penna' : 'Penna__scrittura_', \
                 'vaso' : 'Ceramica', \
                 'quaderno' : 'Quaderno_scolastico', \
                'topo' : 'Mus_musculus'} 

for word in tqdm(it_words):
    
    if word not in bert_mapping.keys():
        capital_word = word.capitalize()
    else:
        capital_word = bert_mapping[word]

    folder_word = capital_word[:3]
    file_path = os.path.join(wiki_files_folder, \
                             folder_word, \
                             '{}.txt'.format(capital_word))

    if not os.path.exists(file_path):
        print(word)
        import pdb; pdb.set_trace()
    assert os.path.exists(file_path)

    with open(file_path) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    lines =[l[0] if l[0] not in [word, capital_word.lower()] and l[1] not in [word, capital_word.lower()] else word \
                                for l in lines]
    lines = [l for l in lines if 'WORD' not in l]

    if lines[-1] != 'BREAK':
        lines.append('BREAK')

    breaks = [0] + [w_i for w_i, w in enumerate(lines) if w == 'BREAK']
    lines = [lines[b+1:breaks[b_i+1]] for b_i, b in enumerate(breaks) if b_i!=len(breaks)-1]
    lines = [l for l in lines if len(l) > 4]

    final_lines = list()
    for l in lines:

        if len(l) < 128:
            line = ' '.join(l)
            final_lines.append(line)

        else:
            
            all_split_points = [0] + [w_i for w_i, w in enumerate(l) if w == '.' or w == ';']
            split_lines = [l[b+1:all_split_points[b_i+1]] for b_i, b in enumerate(all_split_points) if b_i!=len(all_split_points)-1]

            reunited_lines = list()
            starter = split_lines[0]
            for s_l in split_lines[1:]:
                if len(starter + s_l) > 127:
                    reunited_lines.append(starter)
                    starter = s_l
                else:
                    starter = starter + s_l
            for r_l in reunited_lines:
                assert len(r_l) <= 127
                final_lines.append(' '.join(r_l))

    out_file_path = os.path.join(output_folder, '{}.wiki'.format(word))
    with open(out_file_path, 'w') as o:
        for l in final_lines:
            o.write(l)
            o.write('\n') 
