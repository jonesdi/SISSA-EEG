import collections
import os
import matplotlib
import numpy
matplotlib.use('Agg')

import matplotlib.pyplot as plt

### File 2: plotting conditions against one another
def plot_two(s, electrode_name, computational_model, output_path, current_electrode_ps, current_electrode_rhos, current_permutation_rhos, time_points, counts):

    colors = {'wrong' : 'darkorange', 'correct' : 'teal',
             'medium' : 'teal', 'high' : 'darkorange', 'low' : 'darkgray'}

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.1)
    ax.set_xmargin(0.1)
    #ax.invert_yaxis()

    for condition, condition_rhos in current_electrode_rhos.items():

        ### Main line
        xs = [k[0] for k in current_electrode_ps[condition]]
        ax.plot(xs, condition_rhos, label='{} - N={}'.format(condition, counts[condition]), color=colors[condition])

        ### Permutation line and errorbars
        ax.errorbar(x=xs, y=[numpy.nanmean(v) for v in current_permutation_rhos[condition]], yerr=[numpy.nanstd(v) for v in current_permutation_rhos[condition]], label='permutation avg', color='darkgrey', ecolor=colors[condition], alpha=.15)

        ### Significant time points
        xs = [k[0] for k in current_electrode_ps[condition] if k[1] <= .05]
        ys = [condition_rhos[i] for i, k in enumerate(current_electrode_ps[condition]) if k[1] <= .05] 
        ax.scatter(xs, ys, label='p<=.05'.format(condition), edgecolors=colors[condition], linewidths=1., color='white')

    if 'subjective' in output_path:
        ax.legend(ncol=5, loc=9, bbox_to_anchor=(0.5, 1.135), fontsize='x-small')
    else:
        ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.135), fontsize='x-small')

    ax.set_ylabel('Spearman rho')
    ax.set_xlabel('Time')
    ax.set_title('{} - Spearman rho values against {} for subject {} at each time point'.format(electrode_name, computational_model,s), pad=40)
    #ax.hlines(y=1., xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}_plot.png'.format(electrode_name)), dpi=300)
    plt.clf()
    plt.close()

### Plotting the average of all the subjects
def plot_three(s, time_points, y_dict, data_type, output_path, counts={}):

    colors = {'wrong' : 'darkorange', 'correct' : 'teal',
             'medium' : 'teal', 'high' : 'darkorange', 'low' : 'darkgray'}
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.1)
    ax.set_xmargin(0.1)


    for condition, values_lists in y_dict.items():
        if 'average' in data_type:
            xs = time_points
            ys = values_lists
            ax.plot(xs, ys, label=condition, color=colors[condition])
            #ax.scatter(xs, ys, s=5., label=condition, color=colors[condition])
                
        else:
            for electrodes, values in values_lists.items():
                xs = [k[0] for k in values if k[1] <= .05]
                ys = [k[1] for k in values if k[1] <= .05] 
                ax.scatter(xs, ys, s=5., label='{} - N={}'.format(condition, counts[condition]), color=colors[condition])

    model = 'w2v' if 'w2v' in output_path else 'original cooc'
    if not 'average' in data_type:
        ax.set_title('Statistically significant time points for subject {}\nmodel: {}'.format(s, model), pad=40)
        ax.set_ylabel('{}-value'.format(data_type.capitalize()))
    else:
        ax.set_title('Average p-values for subject {} - model: {}'.format(s, model), pad=40)
        ax.set_ylabel('average {}-value'.format(data_type.capitalize()))

    if 'subjective' in output_path:
        ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
    else:
        ax.legend(ncol=2, loc=9, bbox_to_anchor=(0.5, 1.125))

    ax.invert_yaxis()
    ax.set_xlabel('Time')
    ax.hlines(y=.05, xmin=time_points[0], xmax=time_points[-1], linestyle='dashed', color='darkgrey')
    plt.savefig(os.path.join(output_path, '{}_{}-values_plot.png'.format(model, data_type)), dpi=300)
    plt.clf()
    plt.close()

def confusion_matrix(s, electrode, data_type, computational_model, matrix, condition, plot_path):

    fig, ax = plt.subplots(constrained_layout=True)
    
    cmaps = {'correct' : 'Oranges', 'wrong' : 'PuBu', 'high' : 'PuBu', 'medium' : 'Oranges', 'low' : 'Greys'}

    if data_type == 'rho':
        mat = ax.imshow(matrix, cmap=cmaps[condition], extent=(-0.2,1.1,32,0))
        ax.set_title('Significant {}-values over time points\nBundle {} - Subject {} - Condition: {}'.format(data_type.capitalize(), electrode, s, condition.capitalize()), pad=10)
    else:
        new_matrix = list()
        for mat in matrix:
            new_mat = [k if k <= .05 else .5 for k in mat]
            new_matrix.append(new_mat)
        #mat = ax.matshow(new_matrix, cmap='YlOrRd_r')
        mat = ax.imshow(new_matrix, cmap='{}_r'.format(cmaps[condition]), extent=(-0.2,1.1,32,0))
        ax.set_title('{} values against {} over time points\nBundle {} - Subject {} - Condition: {}'.format(data_type.capitalize(), computational_model, electrode, s, condition.capitalize()), pad=10)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
    ax.set_aspect(aspect='auto')
    ax.set_yticks([i-.5 for i in range(1, 33)])
    ax.set_yticklabels(['{}{}'.format(electrode, i) for i in range(1, 33)], fontsize='xx-small')
    ax.hlines(y=[i for i in range(32)], xmin=-0.2, xmax=1.1, color='black', linewidths=.5)
    ax.vlines(x=[float(i)/10 for i in range(-2, 11)], ymin=0, ymax=32, color='darkgray', linewidths=.5, linestyles='dashed')
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    plt.colorbar(mat, ax=ax)
    plt.savefig(os.path.join(plot_path, '{}_bundle_{}_{}_matrix.png'.format(data_type, electrode, condition)), dpi=300)
    plt.clf()
    plt.close()
