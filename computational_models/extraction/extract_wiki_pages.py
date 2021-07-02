import argparse
import os

from tqdm import tqdm

from extraction_utils import read_words

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')
args = parser.parse_args()

it_words, _ = read_words(args)

### Creating the output folder
output_folder = os.path.join('..', '..', 'resources', 'it_wiki_pages')
os.makedirs(output_folder, exist_ok=True)

### Converting words into Wikipedia names

wiki_files_folder = '/import/cogsci/andrea/dataset/corpora/wikipedia_italian/annotated_it_wiki_article_by_article'

bert_mapping = {\
                'cane' : 'Canis_lupus_familiaris', \
                'gatto' : 'Felis_silvestris_catus', \
                'cavallo' : 'Equus_ferus_caballus', \
                'leone' : 'Panthera_leo', \
                'pollo' : 'Gallus_gallus_domesticus', \
                'topo' : 'Mus_musculus', \
                'mucca' : 'Bos_taurus', \
                'tartaruga' : 'Testudines', \
                'piatto' : 'Piatto__stoviglia_', \
                'camera' : 'Stanza__architettura_', \
                'radio' : 'Radio__apparecchio_', \
                'vite' : 'Vite__meccanica_', \
                'furetto' : 'Mustela_putorius_furo', \
                'capra' : 'Capra_hircus', \
                'delfino' : 'Delphinidae', \
                'medusa' : 'Medusa__zoologia_', \
                'agnello' : 'Ovis_aries', \
                'elefante' : 'Elephantidae', \
                'pinguino' : 'Spheniscidae', \
                'foca' : 'Phocidae', \
                'asino' : 'Equus_africanus_asinus', \
                'scimmia' : 'Gorilla', \
                'corvo' : 'Corvus', \
                'leopardo' : 'Panthera_pardus', \
                'mattarello' : 'Matterello', \
                'penna' : 'Penna__scrittura_', \
                'sega' : 'Sega__strumento_', \
                'orso' : 'Ursidae', \
                'gabbiano' : 'Larinae', \
                'maiale' : 'Sus_scrofa_domesticus', \
                'serpente' : 'Serpentes', \
                'tigre' : 'Panthera_tigris', \
                'gufo' : 'Asio_otus', \
                'passero' : 'Passer_domesticus', \
                'tenda' : 'Tenda__arredamento_', \
                'piatto' : 'Piatto__stoviglia_', \
                'scopa' : 'Scopa__strumento_', \
                 'vaso' : 'Ceramica', \
                 'quaderno' : 'Quaderno_scolastico', \
                 }

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
                try:
                    assert len(r_l) <= 127
                except AssertionError:
                    print('One sentence for {} was {} tokens long'.format(word, len(r_l)))
                final_lines.append(' '.join(r_l))

    out_file_path = os.path.join(output_folder, '{}.wiki'.format(word))
    with open(out_file_path, 'w') as o:
        for l in final_lines:
            o.write(l)
            o.write('\n') 
