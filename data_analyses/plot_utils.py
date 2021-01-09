
    print('\nNow plotting the results...')
        
    for certainty, results in subject_results.items():
    
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_ymargin(0.5)
        ax.set_xmargin(0.1)

        all_results = results[:-1]
        results_one = [k[0][1] for k in all_results]
        results_two = [k[1][1] for k in all_results]
        label_one = all_results[0][0][0]
        label_two = all_results[0][1][0]

        number_words = results[-1]

        ax.plot([v for k, v in time_points.items()], results_one, label=label_one)
        ax.plot([v for k, v in time_points.items()], results_two, label=label_two)

        ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.125))
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Time')
        ax.set_title('Correlation with Word2Vec at each time point for subject {}\n{} certainty - {} words considered'.format(s, certainty.capitalize(), number_words), pad=40)

        if args.targets_only:
            word_selection = 'targets_only'
        else:
            word_selection = 'all_words'
        output_path = os.path.join('rsa_results', word_selection, certainty)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'rsa_sub_{:02}_{}.png'.format(s, certainty)))
        plt.clf()
        plt.close()
