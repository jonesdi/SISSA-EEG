import re
import numpy
import scipy
import os
import pandas
import collections
import time
import argparse
import matplotlib
#matplotlib.use('Agg')

from matplotlib import pyplot 

from scipy import stats
from sklearn.metrics import roc_auc_score

'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', required=True, type=str, help='Path where to find the data')
args = parser.parse_args()
'''

data_folder = 'C:/Users/andre/OneDrive - Queen Mary, University of London/conscious_unconscious_processing/raw_files'
time_now = time.strftime('%d_%b_%H_%M', time.gmtime())
base_folder = os.getcwd()
output_folder = os.path.join(base_folder, 'behavioral_results')
os.makedirs(output_folder, exist_ok=True)


### Reading the stimuli file
with open('{}/../useless files/SISSA-EEG_lab/stimuli_final.csv'.format(data_folder)) as i:
    stimuli = [l.strip().split(';')[:3] for l in i.readlines()][1:]

all_subjects_data = list()

### Reading the participants' logs
for s in range(2, 18):

    current_csv_path = os.path.join(data_folder, 'subject{}'.format(s), 'sub-{:02}_events'.format(s))
    sub_list = list()
    for r in range(1,33):
        ### Correcting for naming mistake
        if s == 2:
            r = r - 1
        with open(os.path.join(current_csv_path, 'run_{:02}_events_log.csv'.format(r))) as i:
            current_lines = [l.strip().split(',')[1:] for l in i.readlines()]
        data_info = current_lines[0][:-1]
        data = [[w for w in l][:-1] for l in current_lines[1:]]
        sub_list.extend(data)
    all_subjects_data.append(sub_list)

all_subjects_data = numpy.array(all_subjects_data)

target_mask = lambda array, subject : array[subject,:,1]=='target'
accuracy_mask = lambda array, subject : array[subject,:,3]=='correct'
judgment_mask = lambda array, subject, judgment : array[subject,:,5]==judgment
word_mask = lambda array, subject, word : array[subject,:,0]==word
certainty_mapping = {'1' : 'low', '2' : 'medium', '3' : 'high'}
value_mapping = {'low' : 'certainty', 'medium' : 'certainty', 'high' : 'certainty', 'correct' : '', 'wrong' : ''}

### Collecting accuracy and judgments

current_plot = collections.defaultdict(list)

for s in range(16):

    accuracy_results = accuracy_mask(all_subjects_data, s)
    correct = len([v for v in accuracy_results if v])/accuracy_results.shape[0]
    wrong = len([v for v in accuracy_results if not v])/accuracy_results.shape[0]
    current_plot['correct'].append(correct)
    current_plot['wrong'].append(wrong)

    for j in ['1', '2', '3']:
        current_judgment_mask = judgment_mask(all_subjects_data, s, j)
        current_true = len([v for v in current_judgment_mask if v])/accuracy_results.shape[0]
        certainty_level = certainty_mapping[j]
        current_plot[certainty_level].append(current_true)

### Overall accuracy and judgments

fig, ax = pyplot.subplots(constrained_layout=True)
for i, kv in enumerate(current_plot.items()):
    ax.violinplot(kv[1], positions=[i+1], showmeans=True)
ax.set_xticklabels(['{}\n{}\n\navg={}'.format(k, value_mapping[k], round(numpy.average(v), 2)) for k, v in current_plot.items()])
ax.vlines(x=2.5, ymin=0.0, ymax=1., colors='darkgray', linestyles='dotted')
ax.set_xticks([i+1 for i in range(len(current_plot.keys()))])
ax.set_title('Across-subjects overview of\ncondition distributions', pad=12.0, fontweight='bold', fontsize='xx-large')
#pyplot.show()
#pyplot.savefig('across_subjects_conditions_overview.png', dpi=300)
pyplot.clf()
pyplot.close()

### Word-by-word breakdown - correct/wrong

words = sorted([w[0] for w in stimuli], key=lambda item : len(item))

current_plot = collections.defaultdict(list)

for s in range(16):

    correct_list = list()
    wrong_list = list()
    actually_used_words = list()

    for w in words:

        masked_word = word_mask(all_subjects_data, s, w)
        current_word_array = all_subjects_data[s, :, :][masked_word]
        word_accuracy = current_word_array[:, 3]=='correct'

        try:
            correct_list.append(len([v for v in word_accuracy if v])/word_accuracy.shape[0])
            wrong_list.append(len([v for v in word_accuracy if not v])/word_accuracy.shape[0])
            actually_used_words.append(w)
        except ZeroDivisionError: ### Unused words
            pass

    assert len(correct_list) == 40
    assert len(wrong_list) == 40

    current_plot['correct'].append(correct_list)
    current_plot['wrong'].append(wrong_list)


fig, ax = pyplot.subplots(constrained_layout=True, figsize=(10., 4.8))
for i, kv in enumerate(current_plot.items()):
    i += 1
    ax.violinplot([[kv[1][s][w] for s in range(16)] for w in range(len(actually_used_words))], positions=[v for v in range(i, (len(actually_used_words)+i)*2, 2)][:40], showmeans=True, showextrema=False)
ax.tick_params(axis='x', labelrotation=90., labelbottom=True, labeltop=True)
ax.set_xticks([i+1.5 for i in range(0, len(actually_used_words)*2, 2)])
ax.set_xticklabels(actually_used_words)
ax.set_title('Across-subjects overview of\nper-word accuracy distributions', pad=12.0, fontweight='bold', fontsize='xx-large')
#pyplot.show()
#pyplot.savefig('across_subjects_per_word_overview.png', dpi=600)
pyplot.clf()
pyplot.close()

### MISSING
### Word-by-word breakdown - judgment

### Per-data-division AUC

'''

for analysis in ['mixed', 'objective_accuracy', 'subjective_judgments']:

    for s in range(16):

        if analysis == 'objective_accuracy':

            accuracy_results = accuracy_mask(all_subjects_data, s)
            only_correct = all_subjects_data[s, :, :][accuracy_results]
            import pdb; pdb.set_trace()
            cat = categories[trial[0]]
            true_values.append(cat)




            import pdb; pdb.set_trace()
        correct = len([v for v in accuracy_results if v])/accuracy_results.shape[0]
        wrong = len([v for v in accuracy_results if not v])/accuracy_results.shape[0]
        current_plot['correct'].append(correct)
        current_plot['wrong'].append(wrong)


if accuracy == 1:
    predicted_values.append(cat)
else:
    predicted_values.append(opposites[cat])

split_analysis['unconscious'] = [i for i, k in enumerate(current_results) if k[1] == 1 or (k[1] == 2 and k[0] == 0)]
#split_analysis['semiconscious'] = [i for i, k in enumerate(current_results) if k[1] == 2]
split_analysis['conscious'] = [i for i, k in enumerate(current_results) if k[1] == 3 or (k[1] == 2 and k[0] == 1)]
for k, v in split_analysis.items():
    auc = roc_auc_score([mapping[true_values[i]] for i in v], [mapping[predicted_values[i]] for i in v])


word_colors = []
for w in ['viridis', 'Greys', 'Wistia', 'PuBu', 'prism']:
    current_cmap = plt.get_cmap(w)
    if w == 'viridis':
        word_colors.append([current_cmap(abs(i)) for i in range(-256, 0, int(256/20))])
    else:
        word_colors.append([current_cmap(i) for i in range(0, 256, int(256/18))])
'''

all_words = [] 
all_results = []
both_performances_to_plot = collections.defaultdict(lambda : collections.defaultdict(list))
word_by_word = collections.defaultdict(lambda : collections.defaultdict(list))
trial_by_trial = collections.defaultdict(list)

mapping = {'animal' : 1, 'object' : 0}
categories = {k[0] : k[1] for k in stimuli}
opposites = {'animal' : 'object', 'object' : 'animal'}

### Starting with the analyses
if 1 == 1:
#with open(os.path.join(output_folder, 'behavioural_results.txt'), 'w') as o:
    #for root, direct, files in os.walk(base_folder):
        #if 'subject' in root and 'events' in root:

            #subject_number = int(re.sub('\D', '', re.sub('^.+/|_.+$', '', root)))
    counter = collections.defaultdict(list)
    auc_container = collections.defaultdict(list)
    for subject_number in range(2, 18):

        #current_csv_path = os.path.join(args.data_folder, 'subject{}'.format(subject_number), 'sub-{:02}_events'.format(subject_number))
        #current_csv_path = os.path.join(data_folder, 'sub-{:02}_events'.format(subject_number))
        current_csv_path = os.path.join(data_folder, 'subject{}'.format(subject_number), 'sub-{:02}_events'.format(subject_number))
        results_counter = collections.defaultdict(int)
  
        current_results = list()
        true_values = list()
        predicted_values = list()

        for run in range(1, 33):
            if subject_number == 2:
                run = run - 1
            current_file = pandas.read_csv(os.path.join(current_csv_path, 'run_{:02}_events_log.csv'.format(run)))
            current_trial = []
            for index, outcome in enumerate(current_file['Prediction outcome']):

                if current_file['Group'][index] == 'target':

                    word = str(current_file['Word'][index])
                    if word not in all_words:
                        all_words.append(word)
                    accuracy = 1 if outcome == 'correct' else 0
                    certainty = current_file['Certainty'][index]

                    cat = categories[word]
                    true_values.append(cat)
                    if accuracy == 1:
                        predicted_values.append(cat)
                    else:
                        predicted_values.append(opposites[cat])

                    current_results.append((accuracy, certainty))
                    all_results.append((accuracy, certainty))
                    current_trial.append(accuracy)
                    word_by_word[subject_number][word].append((accuracy, certainty))
                    results_counter[outcome] += 1
            trial_by_trial[subject_number].append([numpy.nanmean(current_trial), numpy.nanstd(current_trial)])

        print(subject_number)
        split_analysis = dict()
        split_analysis['subj\n\nunaware'] = [i for i, k in enumerate(current_results) if k[1] == 1]
        split_analysis['subj\n\nsemiaware'] = [i for i, k in enumerate(current_results) if k[1] == 2]
        split_analysis['subj\n\naware'] = [i for i, k in enumerate(current_results) if k[1] == 3]

        split_analysis['subj+obj\n\naware'] = [i for i, k in enumerate(current_results) if k[1] == 3 or (k[1] == 2 and k[0] == 1)]
        split_analysis['subj+obj\n\nunaware'] = [i for i, k in enumerate(current_results) if k[1] == 1 or (k[1] == 2 and k[0] == 0)]

        counter['obj\n\naware'] = [i for i, k in enumerate(current_results) if k[0] == 1]
        counter['obj\n\nunaware'] = [i for i, k in enumerate(current_results) if k[0] == 0]
        import pdb; pdb.set_trace()

        for k, v in split_analysis.items():
            auc = roc_auc_score([mapping[true_values[i]] for i in v], [mapping[predicted_values[i]] for i in v])
            #correct_guesses = len([k for k in v if k==1]) / len(v)
            #wrong_guesses = len([k for k in v if k==0]) / len(v)
            #try:
                #sensitivity = correct_guesses / wrong_guesses
                #import pdb; pdb.set_trace()
            #except ZeroDivisionError:
                #sensitivity = 'no wrong guesses'
            print('{}\tn={}\t{}'.format(k, len(v), auc))
            counter[k].append(len(v))
            auc_container[k].append(auc)

        counter['subj+obj\n\nunaware'].append(len(split_analysis['subj+obj\n\nunaware']))
        counter['subj+obj\n\naware'].append(len(split_analysis['subj+obj\n\naware']))

    
    ### Plotting AUC
    fig, ax = pyplot.subplots(constrained_layout=True)
    for i, kv in enumerate(auc_container.items()):
        ax.violinplot(kv[1], positions=[i+1], showmeans=True, showextrema=False)
    ax.set_xticklabels(['{}\n\navg={}'.format(k, round(numpy.average(v), 2)) for k, v in auc_container.items()])
    ax.vlines(x=3.5, ymin=0.0, ymax=1., colors='darkgray', linestyles='dotted')
    ax.set_xticks([i+1 for i in range(len(auc_container.keys()))])
    ax.set_title('Across-subjects comparisons of AUC', pad=12.0, fontweight='bold', fontsize='xx-large')
    #pyplot.show()
    #import pdb; pdb.set_trace()
    pyplot.savefig('across_subjects_AUC.png', dpi=300)
    pyplot.clf()
    pyplot.close()

    ### Plotting data
    fig, ax = pyplot.subplots(constrained_layout=True)
    for i, kv in enumerate(counter.items()):
        ax.violinplot(kv[1], positions=[i+1], showmeans=True, showextrema=False)
    ax.set_xticklabels(['{}\n\navg={}'.format(k, round(numpy.average(v), 0)) for k, v in counter.items()])
    ax.vlines(x=5.5, ymin=0.0, ymax=300, colors='darkgray', linestyles='dotted')
    ax.vlines(x=2.5, ymin=0.0, ymax=300, colors='darkgray', linestyles='dotted')
    ax.set_xticks([i+1 for i in range(len(counter.keys()))])
    ax.set_title('Data availability comparison\nwith different splits', pad=12.0, fontweight='bold', fontsize='xx-large')
    #pyplot.show()
    #import pdb; pdb.set_trace()
    pyplot.savefig('across_subjects_splits_comparisons.png', dpi=300)
    pyplot.clf()

    '''
        ### Total guesses
        number_judgments = sum([v for k, v in results_counter.items()])
        percentage_right = round((results_counter['correct'] / number_judgments) * 100, 0)

        ### Breakdown of results

        ### Correct guesses  with low certainties
        mapping = {1 : 'low', 2 : 'medium', 3 : 'high'}
        correct_guesses = {mapping[i] : sum([1 for k in current_results if k[0] == 1 and k[1] == i]) for i in range(1, 4)} 
        wrong_guesses = {mapping[i] : sum([1 for k in current_results if k[0] == 0 and k[1] == i]) for i in range(1, 4)} 
        right_and_wrong = {'right' : correct_guesses, 'wrong' : wrong_guesses}
        

        accuracies = [k[0] for k in current_results]
        certainties = [k[1] for k in current_results]

        o.write('Subject n. {:02}\n\n'.format(subject_number))
        o.write('Percentage of right guesses overall:\n\n\t{}%, out of a total of {} guesses\n\n'.format(percentage_right, number_judgments))
        for acc, acc_results in right_and_wrong.items():
            o.write('Percentage of {} guesses for each certainty level:\n\n'.format(acc))
            for level, guesses in acc_results.items():
                number = sum([v for k, v in acc_results.items()])
                current_level_percentage = round((guesses / number) * 100, 0)
                o.write('\t{} certainty: {}%\n'.format(level, current_level_percentage))
                current_level_percentage_overall = round((guesses / number_judgments) * 100, 0)
                both_performances_to_plot[acc][subject_number].append(current_level_percentage_overall)
            o.write('\n')
        o.write('\n')
        o.write('Results for the 1-sample t-test against a 0.5 random baseline:\n\n')
        for i in range(1, 4):
            current_certainties = [k[0] for k in current_results if k[1] == i]
            t_test_result = scipy.stats.ttest_1samp(current_certainties, popmean=0.5)
            o.write('\t{} certainty: mean: {}, statistic: {}, p-value: {}\n'.format(mapping[i], numpy.nanmean(current_certainties), round(t_test_result[0], 2), t_test_result[1]))
        o.write('\n')

        o.write('Correlation between right/wrong guesses and certainty:\n\n') 
        pearson = stats.pearsonr(accuracies, certainties)
        o.write('\tPearson correlation: {}, p-value: {}\n'.format(round(pearson[0], 2), pearson[1]))
        spearman = stats.spearmanr(accuracies, certainties)
        o.write('\tSpearman correlation: {}, p-value: {}\n\n\n'.format(round(spearman[0], 2), spearman[1]))

with open(os.path.join(output_folder, 'behavioural_results.txt'), 'a') as o:
    overall_median = numpy.nanmedian([k[0] for k in all_results])
    overall_mean = numpy.nanmean([k[0] for k in all_results])
    o.write('Grand average results:\n\nMedian accuracy: {}\nMean accuracy: {}\n\n'.format(overall_median, overall_mean))
    for i in range(1, 4):
        current_certainty = sum([1 for k in all_results if k[1] == i])
        current_percentage = (current_certainty / len(all_results)) * 100
        o.write('Percentage of {} certainty judgments: {}%\n'.format(mapping[i], current_percentage)) 


### Plotting general performances

for acc, performances_to_plot in both_performances_to_plot.items():
    subjects = [k for k in performances_to_plot.keys()]

    low = [v[0] for k, v in performances_to_plot.items()]
    medium = [v[1] for k, v in performances_to_plot.items()]
    high = [v[2] for k, v in performances_to_plot.items()]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ylim(bottom=0, top=110)
    ax.set_xticks(subjects)
    ax.hlines(y=100, xmin=min(subjects)-0.5, xmax=max(subjects)+0.5, linestyle='dashed', color='darkgrey')
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    ax.bar(subjects, low, width=0.4, color='grey', label='Low certainty')
    ax.bar(subjects, medium, bottom=low, width=0.4, color='goldenrod', label='Medium certainty')
    ax.bar(subjects, high, bottom=[medium[i]+low[i] for i in range(len(low))], width=0.4, color='teal', label='High certainty')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Subject')
    ax.set_title('Breakdown of {} results per participant'.format(acc), pad=40)
    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    plt.savefig(os.path.join(output_folder, '{}_accuracies.png'.format(acc)))
    plt.clf()

### Plotting word-by-word accuracies

for subject, word_results in word_by_word.items():
    word_accuracies = []
    for w in all_words:

        current_acc = round((sum([k[0] for k in word_results[w]])) / len(word_results[w]) * 100, 2)
        word_accuracies.append(current_acc)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ylim(bottom=0, top=110)
    ax.set_xticks([i for i in range(len(all_words))])
    ax.set_xticklabels(all_words)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.hlines(y=100, xmin=-1., xmax=len(all_words), linestyle='dashed', color='darkgrey')
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    ax.bar(all_words, word_accuracies, width=0.4, color=word_colors[0])

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Words')
    ax.set_title('Accuracies (percent) per word for participant {}'.format(subject), pad=40)
    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    plt.savefig(os.path.join(output_folder, 'word_accuracies_sub_{:02}.png'.format(subject)))
    plt.clf()
    plt.close()

for subject, word_results in word_by_word.items():
    word_certainties = collections.defaultdict(list)
    for w in all_words:

        for i in range(1, 4):
             
            current_cert = [1 for k in word_results[w] if k[1] == i and k[0] == 1]
            word_certainties[i].append(round((sum(current_cert)) / len(word_results[w]) * 100, 2))

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ylim(bottom=0, top=110)
    ax.set_xticks([i for i in range(len(all_words))])
    ax.set_xticklabels(all_words)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.hlines(y=100, xmin=-1., xmax=len(all_words), linestyle='dashed', color='darkgrey')
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    for cert_level, scores in word_certainties.items():
        if cert_level == 1:
            ax.bar(all_words, scores, width=0.4, color='darkgrey', label='Low certainty')
        elif cert_level == 2:
            ax.bar(all_words, scores, width=0.4, bottom=word_certainties[cert_level-1], color='goldenrod', label='Medium certainty')
        elif cert_level == 3:
            ax.bar(all_words, scores, width=0.4, bottom=[word_certainties[cert_level-2][i] + value for i, value in enumerate(word_certainties[cert_level-1])], color='teal', label='High certainty')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Words')
    ax.set_title('Certainties (percent) per correctly predicted word for participant {}'.format(subject), pad=40)
    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    plt.savefig(os.path.join(output_folder, 'word_certainties_sub_{:02}.png'.format(subject)))
    plt.clf()
    plt.close()

### Plotting trial-by-trial performances

fig, ax = plt.subplots(constrained_layout=True)
ax.set_ylim(bottom=0, top=1.1)
ax.set_xlim(left=0, right=36)
#ax.set_xticks([i for i in rnge(1, 33)])
#ax.set_xticklabels([i for i in ra)
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#ax.hlines(y=1, xmin=-1., xmax=len(all_words), linestyle='dashed', color='darkgrey')
ax.set_ymargin(0.5)
ax.set_xmargin(0.1)

c = 2 
for subject, trial_results in trial_by_trial.items():
    #ax.errorbar(x=[i for i in range(1, 33)], y=[v[0] for v in trial_results], yerr=[v[1]*v[1] for v in trial_results], color=word_colors[2][c])
    #ax.plot(x=[i+((c+8)*0.1) for i in range(1, 33)], y=[v[0] for v in trial_results], s=3, color=word_colors[4][c])
    ax.plot([v[0] for v in trial_results], color=word_colors[4][c])
    c += 1

ax.set_ylabel('Average accuracy')
ax.set_xlabel('Trials')
ax.set_title('Average accuracy per trial for all subjects'.format(subject), pad=40)
#ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
plt.savefig(os.path.join(output_folder, 'trial_by_trial_accuracies.png'))
plt.clf()
plt.close()
for k, v in counter.items():
    print('{}\t{}'.format(k, sum(v)))
'''
