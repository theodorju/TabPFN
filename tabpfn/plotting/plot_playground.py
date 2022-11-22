import pickle
import matplotlib.pyplot as plt
import numpy as np

dataset_name = "Iris"
accs_file_path = 'plots/tabpfn_iris_acc_vs_num_steps.pkl'

fake_dict = {'a': np.random.randint(100, size=5),
             'b': np.random.randint(100, size=5),
             'c': np.random.randint(100, size=5)}

def plot_num_steps_vs_acc(accs_file, num_steps_range,test=True):
    if not test:
        with open(accs_file, 'rb') as f:
            accs = pickle.load(f)
    else:
        accs= fake_dict
    x = list(num_steps_range)
    for key in accs.keys():
        y = accs[key]
        plt.plot(x, y, label=key)
    plt.legend()
    plt.title(f'Number of Adversarial Steps vs Accuracy for {dataset_name}')
    plt.xlabel("Number of steps") ## TODO: only integer steps
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

plot_num_steps_vs_acc(accs_file_path, range(0,20), test=False)
