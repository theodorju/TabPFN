import pickle
import matplotlib.pyplot as plt
import itertools
import numpy as np
datasets = ['breast_cancer','iris']#,'titanic','digits']
lrs = [0.001,0.01,0.1,0.5]
models = ['tabPFN','askl2','autogluon','xgboost','mlp']
accs_file_path = '../results/breast_cancer_results_0.01.pkl'

def plot_norm_vs_acc_diff_models():
    colours_ = ['green', 'blue', 'red', 'magenta', 'purple']
    colours = {k: colours_[i] for i, k in enumerate(models)}
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
            for model_name in models:
                y = results_dict[model_name]['accuracy']
                axs[i,j].plot(x, y, color= colours[model_name], label=model_name)
                axs[i,j].set_title(f"{dataset}, lr = {lr}")
                axs[i,j].grid()

    fig.text(0.5, 0.04, 'L2 norm', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.suptitle('L2 norm every 4 steps vs Accuracy')
    axs[1,1].legend(loc='upper right',bbox_to_anchor=(1.2, 1.05))
    plt.show()


def plot_norm_vs_acc_diff_num_features(lr=0.01):
    colours_ = ['green', 'blue', 'red', 'magenta']
    colours = {k: colours_[i] for i, k in enumerate(datasets)}
    for i, lr in enumerate(lrs):
        for j, dataset in enumerate(datasets):
            accs_file_path = f'../results/{dataset}_results_{lr}.pkl'
            try:
                with open(accs_file_path, 'rb') as f:
                    results_dict = pickle.load(f)
            except Exception:
                continue
            #x = results_dict['tabPFN']['l2_norm_overall']
            x = list(range(0,37,4))
            y = results_dict['tabPFN']['accuracy']
            plt.plot(x, y, color=colours[dataset], label=dataset+'_' + str(lr))
        plt.grid()
        plt.xlabel('L2 norm')
        plt.ylabel('Accuracy')
        plt.title('L2 norm vs Accuracy for different datasets TabPFN')
    plt.legend(loc='upper right')
    plt.show()

#plot_norm_vs_acc_diff_models()
plot_norm_vs_acc_diff_num_features(0.1)
print('here')
