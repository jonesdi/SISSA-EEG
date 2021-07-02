import os

def read_words(args):
    ### Reading the list of it_words

    experiment_path = os.path.join('..', '..', 'lab', \
                                'lab_{}'.format(args.experiment_id))
    if args.experiment_id == 'one':
        file_name = 'stimuli_final.csv'
        separator = ';'
        cat_index = 1
        en_index = 3
    elif args.experiment_id == 'two':
        file_name = 'chosen_words.txt'
        separator = '\t'
        cat_index = 2
        en_index = 1

    stimuli_file = os.path.join(experiment_path, file_name)
    with open(stimuli_file) as i:
        lines = [l.strip().split(separator) \
                   for l in i.readlines()][1:]

    it_words = [l[0] for l in lines]
    en_words = [l[en_index] for l in lines]

    if args.experiment_id == 'one':
        ### Correcting Wordnet names
        en_words = [w.split('_') for w in en_words]
        en_words = ['{}.n.01'.format(w[0].replace(' ', '_')) if len(w)==1 else '{}.n.{}'.format(w[0].replace(' ', '_'), w[1]) for w in en_words]
        
        ### Removing unused words
        ordered_indices = list(range(10)) + list(range(20, 30)) + list(range(10, 20)) + list(range(40, 50))

        it_words = [it_words[i] for i in ordered_indices]
        en_words = [en_words[i] for i in ordered_indices]

    return it_words, en_words
