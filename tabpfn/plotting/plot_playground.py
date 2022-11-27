import pickle
import matplotlib.pyplot as plt
import itertools
import numpy as np
datasets = ['breast_cancer','iris']#,'titanic','digits']
lrs = [0.001,0.01,0.1,0.5]
accs_file_path = '../results/breast_cancer_results_0.01.pkl'

def plot_num_steps_vs_acc():
    fig, axs = plt.subplots(nrows = len(lrs), ncols=len(datasets), figsize=(20, 16))
    for i,lr in enumerate(lrs):
        for j, dataset in enumerate(datasets):
            accs_file_path = f'../results/{dataset}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception:
                continue
            x = results_dict['tabPFN']['l2_norm_overall']
            for model_name in results_dict.keys():
                y = results_dict[model_name]['accuracy']
                axs[i,j].plot(x, y, label=model_name)
                axs[i,j].set_title(f"{dataset}, lr = {lr}")
                axs[i,j].grid()


    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.text(0.5, 0.04, 'L2 norm', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.suptitle('L2 norm every 4 steps vs Accuracy')
    fig.legend(handles, labels, loc='upper right')
    plt.grid(True)
    plt.show()


plot_num_steps_vs_acc()
print('here')
