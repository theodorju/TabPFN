import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np

import plotly.express as px
from scipy import stats
from plotting_globals import spearman_corr_df
from matplotlib.patches import Polygon

fontsizes = {'title': 24, 'label': 16, 'legend_title': 18, 'subplot_title': 18, 'ticklabel': 16}

# SKLearn datasets
datasets_sk = ["breast_cancer"]

# OpenML datasets (tabpfn crashed on 14, 29)
datasets_open = ["11", '16', '22', '37', '14']
# datasets_open_full_results = ['11','16','22','31']

open_ml_name_dict = {
    '11': 'balance-scale',
    '18': 'mfeat-morphological',
    '16': 'mfeat-karhunen',
    '22': 'mfeat-zernike',
    '31': 'credit-g',
    '37': 'diabetes',
    '23': 'cmc',
    '14': 'mfeat-fourier'
}

# Dataset feature mapping
dataset_features_num_dict = {
    'iris': 4,
    '11': 4,
    '18': 6,
    '37': 8,
    'breast_cancer': 10,
    '23': 9,
    'titanic': 12,
    '31': 20,
    '22': 47,
    'digits': 64,
    '16': 64,
    '14': 76,
}

# Merge datasets
datasets = datasets_sk + datasets_open
# datasets = list(dataset_features_num_dict.keys())

# Learning rates used in the experiments
lrs = [0.001, 0.0025, 0.005, 0.01, 0.1]
lrs = [0.005, 0.01]
# Baseline models
models = ['tabPFN', 'askl2', 'autogluon', 'xgboost', 'mlp']

axis_titles = {
    "l2_norm_overall": " L2 norm of (X_attacked - X_original) ",
    "accuracy": "Accuracy",
    "range": "Number of steps"
}

# Results directory
#dir = '../results/'
dir = '../results_l2-poster/'

# Colors used in the plots (chosen to be colorblind friendly)
colours_ = ['red', 'blue', 'green', 'purple', 'orange', 'olive', 'brown', 'hotpink', 'magenta', 'cyan', 'navy', 'teal']


def plot_norm_vs_acc_diff_models(
        range_=None,
        x_axis="l2_norm_overall",
        y_axis="accuracy",
        fig_title="Title"
) -> None:
    """
    Plot L2-Norm vs accuracy for different models. L2-Norm is calculated as the squared distance of the attacked
    features from the original features.

    Args:
        range_ (str or range): (Optional) number of steps to plot. Default None.
        x_axis (str): (Optional) x-axis to plot from results_dict. Default "l2_norm_overall".
        y_axis (str): (Optional) y-axis to plot from results_dict. Default "accuracy".
        fig_title (str): (Optional) title of the figure. Default "Title".

    Returns:
        None
    """

    # Create necessary colors dictionary
    colours = {k: colours_[i] for i, k in enumerate(models)}

    fig, axs = plt.subplots(nrows=len(lrs), ncols=len(datasets), figsize=(30, 10))

    # Loop over learning rates
    for i, lr in enumerate(lrs):

        # Loop over datasets
        for j, dataset_name in enumerate(datasets):

            # Set axes format to 2-floating poins
            axs[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            axs[i, j].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

            # Get results file path
            accs_file_path = f'{dir}{dataset_name}_results_{lr}.pkl'

            # try-except because tabpfn may have crushed for that dataset-lr combination
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                print(f"No results file for dataset: {dataset_name} on learning rate: {lr}.")
                continue

            # If no range is given, l2-norm-overall from results_dict for x-axis
            if range_ is None:
                x = np.round(results_dict['tabPFN'][x_axis], 2)

            # Use accuracy in x-axis
            elif range_ == 'accuracyTab':
                x = np.round(results_dict['tabPFN']['accuracy'], 2)
                min_x = min(x)
                max_x = max(x)

                # L2 norm magnitude
                s = results_dict['tabPFN']['l2_norm_overall']

                # Number of steps
                # s = list(range(0, 101, 4))

                # Log if some baseline model has failed for that dataset-lr combination
                if sum(s) == 0:
                    print(dataset_name, lr)
                    continue

                # Normalize axis
                s_norm = (s - np.min(s)) / (np.max(s) - np.min(s))

            # If a range is provided, use that range for x-axis
            else:
                x = list(range_)

            # Loop over models
            for model_name in models:

                # round to 2 decimal points
                y = np.round(results_dict[model_name][y_axis], 2)

                # If they don't have the same number of results as TabPFN then the competing framework has failed on
                # that dataset-lr combination
                if len(y) != len(x):
                    print(model_name, dataset_name, lr)
                    continue

                # Normalize the plot in case accuracy is used in x-axis to avoid points clustered in one corner of the
                # plot
                if range_ == 'accuracyTab':
                    axs[i, j].scatter(x, y, color=colours[model_name], label=model_name, s=s_norm * 35, alpha=0.5)
                    min_diag = min(min_x, min(y))
                    max_diag = max(max_x, max(y))
                    axs[i, j].plot([min_diag, max_diag], [min_diag, max_diag],
                                   linestyle="--", linewidth=2, color='grey')

                else:
                    axs[i, j].plot(x, y, color=colours[model_name], label=model_name)
                    axs[i, j].axhline(y=0.5, linestyle='--', linewidth=2, color='grey')


            # Display dataset name for the top plots
            if i == 0:
                axs[i, j].set_title(
                    f"{open_ml_name_dict[dataset_name]} (ID {dataset_name})" if dataset_name.isdigit()
                    else f"{dataset_name}", fontsize=18)

            # Display learning rate for the left plots
            if j == len(datasets)-1:
                axs[i, j].set_ylabel(f"lr = {lr}", fontsize=18)
                axs[i, j].yaxis.set_label_position("right")


            # Display grid
            axs[i, j].grid(True)

    # Modify x-axis text
    fig.text(0.55, 0.04, axis_titles[x_axis] if range_ != 'accuracyTab' else 'TabPFN accuracy', ha='center',
             fontsize=22)

    # Modify y-axis text
    fig.text(0.08, 0.5, axis_titles[y_axis] if range_ != 'accuracyTab' else 'Accuracy of Baseline Models', va='center',
             rotation='vertical', fontsize=22)

    # Modify figure title
    fig.text(0.5, 0.94, fig_title, ha='center', fontsize=24)
    handles, labels = fig.axes[0].get_legend_handles_labels()

    # Reposition legend
    legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.62), title='Models',fontsize=16,)
    plt.setp(legend.get_title(), fontsize='18')
    plt.savefig('tabpfn_vs_baselines_l2_norm_dpi_600.png', dpi=600)
    plt.show()


def plot_norm_vs_acc_diff_num_features(
        inner_loop,
        outer_loop,
        outer_is_dataset,
        range_=None,
        l2_norm=False,
        normalize = True,
) -> None:
    """

    Args:
        inner_loop (list): List of datasets or learning rates.
        outer_loop (list): List of datasets or learning rates.
        outer_is_dataset (bool): True if outer_loop is a list of datasets,
            False if outer_loop is a list of learning rates.
        range_ (range): Range of number of features to plot.
        l2_norm (bool): True if l2-norm is used in x-axis, False if number of features is used in x-axis.

    Returns:
        None
    """

    # colours list
    colours_ = list(mcolors.TABLEAU_COLORS.keys()) + ['darkblue', 'darkslategray']
    colours_ = [(0.2,0.3,0.9,0.09*i)  for i,_ in enumerate(datasets)]

    # Populate dictionary with colours and extract number of columns if outer_loop is a list of datasets
    if outer_is_dataset:
        colours = {k: colours_[i] for i, k in enumerate(lrs)}
        n_cols = len(datasets)

    # Populate dictionary with colours and extract number of columns if outer_loop is a list of learning rates
    else:
        colours = {k: colours_[i] for i, k in enumerate(datasets)}
        n_cols = len(lrs)

    # Create figure for l2_norm vs accuracy
    if l2_norm:
        fig, axs = plt.subplots(nrows=2, ncols=n_cols, figsize=(20, 16))

    # Create figure for number of features vs accuracy
    else:
        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(16, 6), sharey="all")

    for i, outer in enumerate(outer_loop):
        for j, inner in enumerate(inner_loop):

            if outer_is_dataset:
                dataset_name = outer
                accs_file_path = f'{dir}{dataset_name}_results_{inner}.pkl'
                sub_plot_label = f'lr:= {inner}'
                sub_plot_title = f"{open_ml_name_dict[dataset_name]} (ID {dataset_name})" if dataset_name.isdigit() \
                    else f"{dataset_name}"

            else:
                dataset_name = inner
                accs_file_path = f'{dir}{dataset_name}_results_{outer}.pkl'
                sub_plot_label = f"{open_ml_name_dict[dataset_name]}(#{dataset_features_num_dict[dataset_name]})" if \
                    dataset_name.isdigit() \
                    else f"{dataset_name}(#{dataset_features_num_dict[dataset_name]})"
                sub_plot_title = f'lr:= {outer}'

            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                continue

            # Get the L2 overall norm for TabPFN. The L2 norm is calculated during experiment execution as
            # the difference of the features at a specific time step of gradient ascent to the initial features.
            if normalize:
                X_original = results_dict['tabPFN']['X_test'][0]
                X_mean = np.mean(X_original)
                X_std = np.std(X_original)
                X_normalized = (X_original - X_mean)/X_std
                l2_norms_normalized = [np.linalg.norm((x_test - np.mean(x_test))/np.std(x_test) - X_normalized) for
                                       x_test in
                                       results_dict['tabPFN']['X_test']]
                #l2_norms_normalized_rescaled =  (l2_norms_normalized - min(l2_norms_normalized)) / (max(
                    #l2_norms_normalized) - min(l2_norms_normalized))
            if range_:
                x_steps = list(range_)
            else:
                x_steps = results_dict['tabPFN']['l1_norm_overall'] if not normalize else l2_norms_normalized
            y = results_dict['tabPFN']['accuracy']

            # Axis formatting based on l2_norm
            if l2_norm:
                axs[0, i].plot(x_steps, y, color=colours[inner], label=sub_plot_label)
                axs[0, i].set_title(sub_plot_title)
                axs[0, i].set_ylabel("accuracy")
                axs[0, i].grid()
                axs[1, i].plot(x_steps, x_norm, color=colours[inner], label=sub_plot_label)
                axs[1, i].set_title(sub_plot_title)
                axs[1, i].set_ylabel("l2 norm")
                axs[1, i].grid()
            else:
                axs[i].plot(x_steps, y, color=colours[inner], label=sub_plot_label)
                axs[i].yaxis.set_tick_params(labelbottom=True)
                axs[i].set_title(sub_plot_title)
                if i ==0:
                    axs[i].set_ylabel('Accuracy')
                axs[i].grid()

        # Text formatting based on l2_norm
        if l2_norm:
            fig.text(0.5, 0.49, 'number of steps', ha='center')
            fig.text(0.5, 0.075, 'number of steps', ha='center')
        else:
            fig.text(0.5, 0.05, 'number of steps' if range_ else 'L1 norm', ha='center')

    handles, labels = fig.axes[0].get_legend_handles_labels()

    # Reposition legend
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_num_steps_vs_norm(range_) -> None:
    """
    Plot the L2 norm of the features at each time step of gradient ascent for TabPFN.
    Args:
        range_ (range): number of steps to plot

    Returns:
        None
    """

    # colours dictionary
    colours = {k: colours_[i] for i, k in enumerate(lrs)}

    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 16), sharey="all")

    # Loop over datasets and learning rates
    for i, dataset_name in enumerate(datasets):
        for j, lr in enumerate(lrs):

            accs_file_path = f'{dir}{dataset_name}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                continue

            x_steps = list(range_)
            y = results_dict['tabPFN']['l2_norm_overall']

            if len(y) != len(x_steps):
                continue

            # Set log scale for y-axis
            axs[i].set_yscale('log')
            axs[i].plot(x_steps, y, color=colours[lr], label=f"lr:= {lr}")
            axs[i].set_title(f"{dataset_name}")
            axs[i].grid()

        # Text formatting
        fig.text(0.5, 0.01, 'number of steps', ha='center')
        fig.text(0.04, 0.5, 'L2 norm Overall', va='center', rotation='vertical')
        fig.suptitle('Distance of attacked features from original features per step')

    handles, labels = fig.axes[0].get_legend_handles_labels()

    # Reposition legend
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_norm_vs_acc_all_datasets() -> None:
    """
    Plot the accuracy of TabPFN and any successfully executed pipelines against the L2 norm of the features at each time
    step of gradient ascent for TabPFN.

    Returns:
        None
    """

    # Colors dictionary
    colours = {k: colours_[i] for i, k in enumerate(datasets)}
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharey="all")

    # Loop over learning rates and datasets
    for i, lr in enumerate(lrs):
        for j, dataset_name in enumerate(datasets):

            accs_file_path = f'{dir}{dataset_name}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                continue

            # The experiments were executed for 100 steps of gradient ascent with the results being saved every 4 steps.
            x_steps = list(range(0, 101, 4))

            # Load results for TabPFN
            y = results_dict['tabPFN']['accuracy']

            # If execution on TabPFN was not successful for the current dataset-lr combination, skip
            if len(y) != len(x_steps):
                continue
            axs.plot(x_steps, y, color=colours[dataset_name], label=f"{dataset_name}")
            axs.set_title(f"lr:={lr}")
            axs.grid(True)

        # Text modifications
        fig.text(0.5, 0.01, 'L2 norm Overall', ha='center')
        fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')

        fig.suptitle('Performance of TabPFN on Different Attacked Datasets ')

    handles, labels = fig.axes[0].get_legend_handles_labels()

    # Reposition legend
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_tab_no_attack() -> None:
    """
    Plot the accuracy against the baselines on the original datasets.

    Returns:
        None
    """

    # Colors dictionary
    colours = {k: colours_[i] for i, k in enumerate(models)}

    _, _ = plt.subplots(nrows=len(lrs), ncols=len(datasets), figsize=(20, 16))

    for j, dataset_name in enumerate(datasets):

        # Learning rate does not matter here since we are plotting at step zero (no attack yet)
        accs_file_path = f'{dir}{dataset_name}_results_{0.005}.pkl'
        try:
            with open(accs_file_path, 'rb') as f:
                results_dict = pickle.load(f)

        except Exception:
            continue

        # Get the accuracy of TabPFN before any attack takes place
        x = results_dict['tabPFN']['accuracy'][0]

        # Accuracies of baselines
        for i, model_name in enumerate(models):
            y = results_dict[model_name]['accuracy'][0]
            print(f"{model_name}: {np.round(y, 2)}")

            if j == 0:
                plt.scatter([x], [y], color=colours[model_name], label=model_name)
            else:
                plt.scatter([x], [y], color=colours[model_name])
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color='grey')

    plt.legend(loc='upper right', bbox_to_anchor=(0.96, 0.96))
    plt.title('TabPFN Accuracy vs Other Models')
    plt.show()

def box_plot_diff_lr():
    data = []
    datasets_taken = []
    for j, dataset_name in enumerate(datasets):
        data_norms = []
        data_accs = []
        for i, lr in enumerate(lrs):
            accs_file_path = f'{dir}{dataset_name}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                continue
            l1_norm = results_dict['tabPFN']['l1_norm_overall'][-1]
            data_norms.append(l1_norm)
            acc = results_dict['tabPFN']['accuracy'][-1]
            data_accs.append(acc)
        if len(data_norms)==0:
            continue
        name = f"{open_ml_name_dict[dataset_name]}" if dataset_name.isdigit() else f"{dataset_name}"
        x_label_title_1 = name+'\nL1-Norm\n('+str(dataset_features_num_dict[dataset_name])+')'
        x_label_title_2 = name+'\nAcc.\n('+str(dataset_features_num_dict[dataset_name])+')'
        datasets_taken.extend([x_label_title_1,x_label_title_2])
        data_norms = (data_norms - min(data_norms)) / (max(data_norms) - min(data_norms))
        data.extend([data_norms,data_accs])

    fig, ax1 = plt.subplots(figsize=(24, 10))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
    )

    # ax1.set_title('Average Distance per Feature Attack and Accuracy for Different lrs', fontsize=24)
    fig.text(0.5, 0.94, 'Average Feature Distance and Average Accuracy for Different LRs', ha='center', fontsize=24)
    # ax1.set_xlabel('Datasets', fontsize=22)
    fig.text(0.5, 0.05, 'Datasets', ha='center', fontsize=22)
    ax1.set_ylabel('Value', fontsize=22)
    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = np.max(data)
    bottom = np.min(data)
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(datasets_taken,
                        rotation=45, fontsize=fontsizes['ticklabel'])

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=18,
                 weight=weights[k], color=box_colors[k])
    # Finally, add a basic legend
    fig.text(0.80, 0.08, f'Avg distance per feature after 100 steps',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             fontsize=16)
    fig.text(0.80, 0.045, 'Accuracy after 100 steps for different lrs',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', fontsize=16)
    fig.text(0.80, 0.0055, '*', color='white', backgroundcolor='silver',
             weight='roman', fontsize=16)
    fig.text(0.815, 0.01, ' Average Value', color='black', weight='roman',
             fontsize=16)
    plt.savefig('abs_distance_vs_accuracy_different_lrs_dpi_600.png', dpi=600)
    plt.show()

def corr_between_features_breakability(scatter=True) -> None:
    """
    Create a horizontal bar plot showing the negative correlation and p-values between the features and the drop in
    accuracy on TabPFN. The datasets are ordered in ascending number of features.

    Returns:
        None
    """

    # Modify font size
    font = {'size': 24}
    matplotlib.rc('font', **font)

    # Datasets containing 4 sklearn and 8 OpenML datasets (detailed experiments were executed on 4 OpenML datasets,
    # here we included 4 more to try and draw conclusions between L2-norm and accuracy drop correlation)
    dataset_ = datasets
    corrs = []
    pvals = []

    for i, dataset_name in enumerate(dataset_):
        distances = []
        accs = []

        for j, lr in enumerate(lrs):

            accs_file_path = f'{dir}{dataset_name}_results_{lr}.pkl'

            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)

            except Exception:
                continue

            distances.extend(results_dict['tabPFN']['l2_norm_overall'])
            accs.extend(results_dict['tabPFN']['accuracy'])

        rho, p_val = stats.spearmanr(distances, accs)
        corrs.append(rho)
        pvals.append(p_val)
        print(f'Dataset: {dataset_name}, #Features:{dataset_features_num_dict[dataset_name]}, '
              f'Corr:{rho}, pval: {p_val}')

    fig, ax = plt.subplots(figsize=(24, 18))
    y_axis = corrs
    if scatter:
        x_axis = [dataset_features_num_dict[dataset_name] for dataset_name in dataset_]
        plt.scatter(x_axis,y_axis)
        plt.xlabel('Number of Features')
        plt.ylabel('Spearman Correlation Score')
    else:
        x_axis = [f"{dataset_name}\n({dataset_features_num_dict[dataset_name]})" for dataset_name in dataset_]
        bars = ax.barh(x_axis, y_axis, color="#004a97", alpha=0.7)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel("Datasets (#features)")
        ax.set_xlabel("Spearman Correlation Score")
        ax.set_xlim([-0.4, -1.05])

        # To get data labels
        for i, bar in enumerate(bars):
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(-0.65, label_y, s='pval = {:.2E}'.format(pvals[i]), color="white")
    plt.show()


def features_vs_corr(only_cat=False) -> None:
    """
    Create an interactive bar chart showing the correlation between the number of features and the drop in accuracy on
    TabPFN using plotly. Features are divided based on their type.

    Returns:
        None
    """

    # Load spearman correlation (manually created dataframe extracted from our experiment runs)
    df = spearman_corr_df

    # Cast number of features to int and correlation to float
    df['val'] = df['val'].astype(int)
    df['corr'] = df['corr'].astype('float')
    if only_cat:
        df = df[df['type'] == 'numerical']
    else:
        df = df[df['type'] =='features']
    x_axis = df['val'].values
    y_axis = df['corr'].values
    rho, p_val = stats.spearmanr(x_axis, y_axis)
    print(rho,p_val)

    plt.scatter(x_axis,y_axis, alpha=0.8)
    plt.ylim([0.7,1])
    plt.xlabel('Number of Features' if not only_cat else 'Number of Categorical Features')
    plt.ylabel('Negative Spearman Correlation Score')
    plt.grid(True)
    plt.show()


# plot_num_steps_vs_norm(range(0, 101, 4))
#
plot_norm_vs_acc_diff_models(range_='accuracyTab',
                             x_axis="l2_norm_overall",
                             y_axis="accuracy",
                             fig_title="TabPFN vs Baselines")

# plot_norm_vs_acc_diff_num_features(outer_loop=lrs,
#                                     inner_loop=datasets,
#                                     outer_is_dataset=False,
#                                     l2_norm=False,
#                                     normalize=True)
# plot_tab_no_attack()
#
# corr_between_features_breakability()
#
# plot_norm_vs_acc_all_datasets()
#
#features_vs_corr(only_cat=True)
# box_plot_diff_lr()
