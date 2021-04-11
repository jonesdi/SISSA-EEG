### Reading lexvar
with open('lexvar.csv') as i:
    lexvar = [l.strip().split(',') for l in i.readlines()]

heading = list()
for index, metric_measure in enumerate(zip(lexvar[0], lexvar[1])):
    metric = metric_measure[0]
    measure = metric_measure[1]
    if metric == '':
        for i in range(index, -1, -1):
            if lexvar[0][i] != '':
                metric = lexvar[0][i]
                break
    value = '{}_{}'.format(metric, measure)
    heading.append(value)

    
chosen_variables = ['WORD_Italian', '   FAM_mean', 'IMAG_mean', 'CONC_mean','Adult WrtFQ_ILC', 'Adult WrtFQ_CoLFIS', 'Adult_NSIZE', 'Adult_BIGR', 'Adult_SYL', 'Lexical_LET']
relevant_indices = [i[0] for i in enumerate(heading) if i[1] in chosen_variables]

lexvar = [[l[i] for i in relevant_indices] for l in lexvar if l[0].lower() in [w[0] for w in stimuli]]
