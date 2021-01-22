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

    colors = {'wrong' : 'goldenrod', 'correct' : 'teal'}
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)
    ax.invert_yaxis()

    for condition, values_lists in y_dict.items():
        if 'average' in data_type:
            '''
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
            '''
            xs = time_points
            ys = values_lists
            ax.scatter(xs, ys, s=5., label=condition, color=colors[condition])
                
        else:
            for electrodes, values in values_lists.items():
                xs = [k[0] for k in values if k[1] <= .05]
                ys = [k[1] for k in values if k[1] <= .05] 
                ax.scatter(xs, ys, s=5., label='{} - N={}'.format(condition, counts[condition]), color=colors[condition])

                #ax.plot(time_points, values, label=condition)

    ax.legend(ncol=2, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('{}-value'.format(data_type.capitalize()))
    ax.set_xlabel('Time')
    ax.set_title('Statistically significant time points for subject {}'.format(s), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}-values_plot.png'.format(data_type)))
    plt.clf()
    plt.close()

def line_and_scatter_plot_all_electrodes_subject_p_values(s, time_points, p_values, original_rhos, permutation_rhos, condition, output_path, electrode_name, counts):
    
    colors = {'wrong' : 'goldenrod', 'correct' : 'teal'}
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)
    #ax.invert_yaxis()

    ax.plot(time_points, original_rhos, label='original - N={}'.format(counts[condition]), color=colors[condition])

    ax.errorbar(x=time_points, y=[numpy.nanmean(v) for v in permutation_rhos], yerr=[numpy.nanstd(v) for v in permutation_rhos], label='permutation avg', color='darkgrey', ecolor='gainsboro')
    #ax.plot(time_points, [numpy.nanmean(v) for v in permutation_rhos], label='permutation avg', color='darkgrey')

    xs = [k[0] for k in p_values if k[1] <= .05]
    ys = [original_rhos[i] for i, k in enumerate(p_values) if k[1] <= .05] 
    ax.scatter(xs, ys, label='p <= 0.05'.format(condition), edgecolors='black', linewidths=1., color='white')

    #ax.plot(time_points, values, label=condition)

    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('Spearman rho')
    ax.set_xlabel('Time')
    ax.set_title('Spearman rho values for subject {} at each time point\nCondition: {}'.format(s, condition.capitalize()), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}_{}_plot.png'.format(electrode_name, condition)))
    plt.clf()
    plt.close()

def subject_electrodes_scatter_plot(s, plot_time_points, subject_electrodes_plot, condition, plot_path):

    fig, ax = plt.subplots(constrained_layout=True)
    #ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    average = collections.defaultdict(int)
    for electrode, t_points in subject_electrodes_plot.items():
       
        ys = [electrode for t in t_points]
        ax.scatter(t_points, ys, s=5.)
        '''
        for t in t_points:
            average[t] += 1
    ys = [average[t] for t in plot_time_points]
    ax.bar(plot_time_points, ys, width = 0.005)
    '''

    ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    ax.set_ylabel('Electrodes')
    ax.set_xlabel('Time')
    ax.set_title('Significant time points for subject {}\nCondition: {}'.format(s, condition.capitalize()), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(plot_path, 'all_electrodes_{}_plot.png'.format(condition)))
    plt.clf()
    plt.close()

def confusion_matrix(s, data_name, matrix, rows_labels, columns_labels, condition, plot_path):

    fig, ax = plt.subplots(constrained_layout=True)
    if 'rho' in data_name:
        #mat = ax.matshow(matrix, cmap='YlOrRd')
        mat = ax.matshow(matrix, cmap='Oranges', extent=(-0.2,1.1,32,0))
    else:
        new_matrix = list()
        for mat in matrix:
            new_mat = [k if k <= .05 else .5 for k in mat]
            new_matrix.append(new_mat)
        #mat = ax.matshow(new_matrix, cmap='YlOrRd_r')
        mat = ax.matshow(new_matrix, cmap='Oranges_r', extent=(-0.2,1.1,32,0))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
    ax.set_aspect(aspect='auto')
    plt.colorbar(mat, ax=ax)
    #ax.set_yticklabels(['']+rows_labels)
    #ax.set_xticklabels(['']+columns_labels)
    ax.set_title('Electrode {} over time points\nSubject {}    Condition: {}'.format(data_name.replace('_', ' - '), s, condition.capitalize()), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(plot_path, '{}_matrix_{}_plot.png'.format(data_name, condition)))
    plt.clf()
    plt.close()
