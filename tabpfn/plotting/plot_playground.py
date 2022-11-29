import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import *
import itertools
import numpy as np
datasets = [load_iris, load_breast_cancer, load_digits, None]
lrs = [0.001, 0.0025, 0.005, 0.01, 0.1]
models = ['tabPFN','askl2','autogluon','xgboost','mlp']
accs_file_path = '../results/breast_cancer_results_0.01.pkl'
axis_titles = {"l2_norm_overall":" L2 norm of (X_attacked - X_original) ",
               "accuracy":"Accuracy",
               "range":"Number of steps"
               }
def plot_norm_vs_acc_diff_models(range = None,x_axis = "l2_norm_overall",y_axis = "accuracy", fig_title="Insert Title"):
    colours_ = ['green', 'blue', 'red', 'magenta', 'purple']
    colours = {k: colours_[i] for i, k in enumerate(models)}
    fig, axs = plt.subplots(nrows = len(lrs), ncols=len(datasets), figsize=(20, 16))
    for i,lr in enumerate(lrs):
        for j, dataset in enumerate(datasets):
            dataset_name = 'titanic' if dataset is None else "_".join(dataset.__name__.split("_")[1:])
            accs_file_path = f'../results/{dataset_name}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception:
                continue
            if range is None:
                x = results_dict['tabPFN'][x_axis]
            else:
                x = list(range)

            for model_name in models:
                y = results_dict[model_name][y_axis]
                if len(y) == 0:
                    continue
                axs[i,j].plot(x, y, color= colours[model_name], label=model_name)
                axs[i,j].axhline(y=0.5,linestyle = '--',linewidth=2, color='grey')
                axs[i,j].set_title(f"{dataset_name}, lr = {lr}")
                axs[i,j].grid()

    fig.text(0.5, 0.04, axis_titles[x_axis], ha='center')
    fig.text(0.04, 0.5, axis_titles[y_axis], va='center', rotation='vertical')
    fig.suptitle(fig_title)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_norm_vs_acc_diff_num_features(inner_loop,outer_loop,outer_is_dataset,range=None,l2_norm = False):
    colours_ = ['green', 'blue', 'red', 'magenta','purple']
    if outer_is_dataset:
        colours = {k: colours_[i] for i, k in enumerate(lrs)}
        n_cols = len(datasets)
    else:
        colours = {k: colours_[i] for i, k in enumerate(datasets)}
        n_cols = len(lrs)

    if l2_norm:
        fig, axs = plt.subplots(nrows = 2, ncols=n_cols, figsize=(20, 16))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(16,8),sharey=True)
    for i, outer in enumerate(outer_loop):
        for j, inner in enumerate(inner_loop):
            if outer_is_dataset:
                dataset_name = 'titanic' if outer is None else "_".join(outer.__name__.split("_")[1:])
                accs_file_path = f'../results/{dataset_name}_results_{inner}.pkl'
                sub_plot_label = f'lr:= {inner}'
                sub_plot_title = f'{dataset_name}'
            else:
                dataset_name = 'titanic' if inner is None else "_".join(inner.__name__.split("_")[1:])
                accs_file_path = f'../results/{dataset_name}_results_{outer}.pkl'
                sub_plot_label = f'{dataset_name}'
                sub_plot_title = f'lr:= {outer}'

            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception:
                continue
            x_norm = results_dict['tabPFN']['l2_norm_overall']
            x_steps = list(range)
            y = results_dict['tabPFN']['accuracy']
            if l2_norm:
                axs[0,i].plot(x_steps, y, color=colours[inner], label=sub_plot_label)
                axs[0,i].set_title(sub_plot_title)
                axs[0,i].grid()
                axs[1, i].plot(x_norm, y, color=colours[inner], label= sub_plot_label)
                axs[1, i].set_title(sub_plot_title)
                axs[1, i].grid()
            else:
                axs[i].plot(x_steps, y, color=colours[inner], label=sub_plot_label)
                axs[i].yaxis.set_tick_params(labelbottom=True)
                axs[i].set_title(sub_plot_title)
                axs[i].grid()
        if l2_norm:
            fig.text(0.5, 0.49, 'number of steps', ha='center')
            fig.text(0.5, 0.075, 'L2 norm', ha='center')
        else:
            fig.text(0.5, 0.05, 'number of steps', ha='center')
        fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
        fig.suptitle('L2 norm vs Accuracy for different datasets (/number of features) TabPFN')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_num_steps_vs_norm(range):
    colours_ = ['green', 'blue', 'red', 'magenta','purple']
    colours = {k: colours_[i] for i, k in enumerate(lrs)}
    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(20, 16),sharey=True)
    for i, dataset in enumerate(datasets):
        for j, lr in enumerate(lrs):
            dataset_name = 'titanic' if dataset is None else "_".join(dataset.__name__.split("_")[1:])
            accs_file_path = f'../results/{dataset_name}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception:
                continue
            x_steps = list(range)
            y = results_dict['tabPFN']['l2_norm_overall']
            if len(y) != len(x_steps):
                continue
            axs[i].set_yscale('log')
            axs[i].plot(x_steps, y, color=colours[lr], label=f"lr:= {lr}")
            axs[i].set_title(f"{dataset_name}")
            axs[i].grid()


        fig.text(0.5, 0.01, 'number of steps', ha='center')
        fig.text(0.04, 0.5, 'L2 norm Overall', va='center', rotation='vertical')
        fig.suptitle('Distance of attacked features from original features per step')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()

#plot_num_steps_vs_norm(range(0,101,4))
#plot_norm_vs_acc_diff_models(range=None,x_axis="l2_norm_overall", y_axis="accuracy", fig_title="L2 norm vs accuracy")
plot_norm_vs_acc_diff_num_features(outer_loop=lrs,inner_loop=datasets,outer_is_dataset=False,range=range(0,101,4), \
                                                                                              l2_norm=False)
print('here')
