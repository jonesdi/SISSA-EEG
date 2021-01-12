import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def basic_line_plot_searchlight_electrodes(time_points, y_dict, condition, model_name, number_of_words, output_path):
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ymargin(0.5)
    ax.set_xmargin(0.1)

    for electrode, values in y_dict.items():
        ax.plot(time_points, values)

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
