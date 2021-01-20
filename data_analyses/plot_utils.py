import collections
import os
import matplotlib
import numpy
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def basic_line_plot_searchlight_electrodes(time_points, y_dict, condition, model_name, number_of_words, output_path):
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    for electrode, values in y_dict.items():
        #xs = [k[0] for k in values]
        #ys = [k[1] for k in values] 
        ax.plot(time_points, values)
        #ax.plot(xs, ys)

    #all_results = results[:-1]
    #results_one = [k[0][1] for k in all_results]
    #results_two = [k[1][1] for k in all_results]
    #label_one = all_results[0][0][0]
    #label_two = all_results[0][1][0]

    #number_words = results[-1]

    #ax.plot([v for k, v in time_points.items()], results_one, label=label_one)
    #ax.plot([v for k, v in time_points.items()], results_one, label=label_one)

    #ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Time')
    #ax.set_title('Correlation with {} at each time point\n{} certainty - {} words considered'.format(s, certainty.capitalize(), number_words), pad=40)
    ax.set_title('Correlation with {} at each time point\nWords used: {}'.format(model_name, len(number_of_words)), pad=20)
    ax.hlines(y=0.0, xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')

    #if args.targets_only:
        #word_selection = 'targets_only'
    #else:
        #word_selection = 'all_words'
    #output_path = os.path.join('rsa_results', word_selection, certainty)
    #os.makedirs(output_path, exist_ok=True)
    #plt.savefig(os.path.join(output_path, 'rsa_sub_{:02}_{}.png'.format(s, certainty)))
    plt.savefig(os.path.join(output_path, '{}.png'.format(condition)))
    plt.clf()
    plt.close()

def basic_line_plot_all_electrodes_subject_p_values(s, time_points, y_dict, data_type, output_path):

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    for condition, values_dict in y_dict.items():
        if 'average' in data_type:
            ax.plot(time_points, values_dict, label=condition)
        else:
            for electrodes, values in values_dict.items():
                ax.plot(time_points, values, label=condition)

    ax.legend(ncol=2, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('T-value')
    ax.set_xlabel('Time')
    ax.set_title('P-values for subject {} at each time point'.format(s), pad=40)
    ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}-values_plot.png'.format(data_type)))
    plt.clf()
    plt.close()

def basic_scatter_plot_all_electrodes_subject_p_values(s, time_points, y_dict, data_type, output_path, counts={}):

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)
    ax.invert_yaxis()

    for condition, values_lists in y_dict.items():
        if 'average' in data_type:
            #ax.plot(time_points, values_dict, label=condition)
            #for subject, values in enumerate(values_lists):
                #subject += 3
            averaged = collections.defaultdict(list)
            xs = [k[0] for values in values_lists for k in values]
            ys = [k[1] for values in values_lists for k in values] 
            for k, v in zip(xs, ys):
                averaged[k].append(v)
            averaged = {k : numpy.nanmean(v) for k, v in averaged.items()}
            xs = [k for k in averaged.keys()]
            ys = [v for k, v in averaged.items()]
            ax.scatter(xs, ys, label=condition)
                
                
        else:
            for electrodes, values in values_lists.items():
                xs = [k[0] for k in values if k[1] <= .05]
                ys = [k[1] for k in values if k[1] <= .05] 
                ax.scatter(xs, ys, label='{} - N={}'.format(condition, counts[condition]))

                #ax.plot(time_points, values, label=condition)

    ax.legend(ncol=2, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('{}-value'.format(data_type.capitalize()))
    ax.set_xlabel('Time')
    ax.set_title('{}-values for subject {} at each time point'.format(data_type.capitalize(), s), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}-values_plot.png'.format(data_type)))
    plt.clf()
    plt.close()

def line_and_scatter_plot_all_electrodes_subject_p_values(s, time_points, y_dict, original_rhos, permutation_rhos, condition, output_path, counts):
    
    colors = {'wrong' : 'goldenrod', 'correct' : 'teal'}
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)
    #ax.invert_yaxis()

    original_rhos = original_rhos[condition]
    ax.plot(time_points, original_rhos, label='original - N={}'.format(counts[condition]), color=colors[condition])

    permutation_rhos = permutation_rhos[condition]
    #ax.errorbar(x=time_points, y=[numpy.nanmean(v) for v in permutation_rhos], yerr=[numpy.nanstd(v) for v in permutation_rhos], label='permutation', color='darkgrey')
    ax.plot(time_points, [numpy.nanmean(v) for v in permutation_rhos], label='permutation', color='darkgrey')

    for electrodes, values in y_dict[condition].items():
        xs = [k[0] for k in values if k[1] <= .05]
        ys = [original_rhos[i] for i, k in enumerate(values) if k[1] <= .05] 
        ax.scatter(xs, ys, label='p <= 0.05'.format(condition), edgecolors='black', alpha=0)

        #ax.plot(time_points, values, label=condition)

    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('Spearman rho')
    ax.set_xlabel('Time')
    ax.set_title('{} - Spearman rho values for subject {} at each time point'.format(condition.capitalize(), s), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}_plot.png'.format(condition)))
    plt.clf()
    plt.close()
