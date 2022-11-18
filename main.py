from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class AdversarialTabPFN:
    def __init__(
            self,
            X_full=None,
            y_full=None,
            datasets_fn=None,
            lmbd=0.1,
            random_state=42,
            test_percentage=0.45,
            optimizer=optim.Adam,
            num_steps=250
    ):
        self.dataset_fn = datasets_fn if datasets_fn is not None else load_breast_cancer
        self.X_full = X_full
        self.y_full = y_full
        self.lmbd = lmbd
        self.random_state = random_state
        self.test_percentage = test_percentage
        self.manual = False
        self.optimizer = optimizer
        self.num_steps = num_steps

    def adversarial_attack(self):
        X_train, X_test, X_test_clean, y_train, y_test = \
            self.load_dataset(self.dataset_fn, self.X_full, self.y_full)

        classifier = TabPFNClassifier(device='cpu')
        classifier.fit(X_train, y_train)

        y_eval_before, p_eval_before, X_full_with_grad = \
            classifier.predict(X_test, y_test, optimizer=self.optimizer, num_steps=self.num_steps,
                               return_winning_probability=True)

        acc = accuracy_score(y_test, y_eval_before)
        print(f"accuracy: {acc}")
        return acc

    def load_dataset(self, dataset_fn=None, X=None, y=None):
        # load dataset
        if dataset_fn:
            X, y = dataset_fn(return_X_y=True)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_percentage, random_state=self.random_state)

        # keep test set before modifications
        X_test_clean = X_test.copy()

        return X_train, X_test, X_test_clean, y_train, y_test


datasets_fn = [load_iris, load_breast_cancer, load_wine, load_digits]
optim = optim.Adam

plot = False
if plot:
    accs = []
    for i in range(0, 20, 2):
        obj = AdversarialTabPFN(datasets_fn=load_iris, optimizer=optim, num_steps=i)
        new_acc = obj.adversarial_attack()
        accs.append(new_acc)
    plt.plot(range(0, 20, 2), accs)
    plt.show()

else:
    obj = AdversarialTabPFN(datasets_fn=load_iris, optimizer=optim, num_steps=2)
    new_acc = obj.adversarial_attack()
