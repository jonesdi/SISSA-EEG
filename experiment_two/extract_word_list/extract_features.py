
### Properties file
with open('13428_2012_291_MOESM5_ESM.txt', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
relevant_columns = ['CONCEPT (IT)', 'FEATURE (IT)', 'Prod_Fr', 'Distinctiveness']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
data = [[l[i] for i in indices] for l in lines[1:]]

words = list(set([d[0] for d in data]))
feature_dict = {w : list() for w in words}

for w in words:
    for d in data:
        if d[0] == w:
            feature_dict[w].append([d[1], float(d[2])*float(d[3])])

### Reading experiment words

with open('chosen_words.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]

words = [l[0] for l in lines]
exp_features = dict()
missing_words = list()

for w in words:
    if w in feature_dict.keys():
        exp_features[w] = feature_dict[w]
    else:
        missing_words.append(w)

### Reading the other features

with open('concepts-features_it.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
relevant_columns = ['Concept(IT)', 'Prod.Frequency', 'Feature', 'Distinctiveness']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
data = [[l[i] for i in indices] for l in lines[1:]]

### Reading the translator from English to Italian
with open('production-data_it.txt', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
lines = [[w for w in l if w != ''] for l in lines] #correcting a bug in the dataset
relevant_columns = ['Feature', 'Phrase']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
translator = {l[indices[0]] : l[indices[1]] for l in lines}

for w in missing_words:
    assert w not in feature_dict.keys()
    feature_dict[w] = list()
    for d in data:
        if d[0] == w:
            feature_dict[w].append([translator[d[2]], float(d[1])*float(d[3])])

for w in missing_words:
    if w in feature_dict.keys():
        exp_features[w] = feature_dict[w]
    else:
        print('problem with {}'.format(w))

exp_features = {k : sorted(v, key=lambda item : item[1], reverse=True) for k, v in exp_features.items()}
exp_features = {k : [val[0] for val in v] for k, v in exp_features.items()}

final_features = {k : list() for k in exp_features.keys()}

for k, v in exp_features.items():
    for feat in v:
        counter = 0
        for k_two, v_two in exp_features.items():
            if k_two != k:
                if feat in v_two:
                    counter += 1
        if counter < 2:
            final_features[k].append(feat)

with open('chosen_features.txt', 'w') as o:
    o.write('Word\tFeatures\n')
    for w in words:
        o.write('{}\t'.format(w))
        for feat in exp_features[w]:
            o.write('{}\t'.format(feat.replace(' ', '_')))
        o.write('\n')
