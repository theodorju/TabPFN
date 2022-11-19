from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
import torch.optim as optim


class AdversarialTabPFN:
    def __init__(
            self,
            X_full=None,
            y_full=None,
            datasets_fn=None,
            lr=0.1,
            random_state=42,
            test_percentage=0.2,
            optimizer=optim.Adam,
            num_steps=250
    ):
        self.dataset_fn = datasets_fn
        self.X_full = X_full
        self.y_full = y_full
        self.lr = lr
        self.random_state = random_state
        self.test_percentage = test_percentage
        self.manual = False
        self.optimizer = optimizer
        self.num_steps = num_steps

        if datasets_fn is None and (X_full is None and y_full is None):
            print("No dataset or X, y provided. Using Breast Cancer Dataset from sklearn.")
            self.dataset_fn = load_breast_cancer

    def adversarial_attack(self):
        X_train, X_test, X_test_clean, y_train, y_test = \
            self.load_dataset(self.dataset_fn, self.X_full, self.y_full)

        classifier = TabPFNClassifier(device='cpu')
        classifier.fit(X_train, y_train)
        y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

        before_acc = accuracy_score(y_test, y_eval)
        print(f"Accuracy: {before_acc}")

        y_eval_before, p_eval_before, X_full_with_grad, X_test_tensor = \
            classifier.predict_attack(X_test, y_test,
                                      optimizer=self.optimizer,
                                      lr=self.lr,
                                      num_steps=self.num_steps,
                                      return_winning_probability=True
                                      )

        acc = accuracy_score(y_test, y_eval_before)
        print(f"Accuracy: {acc}")

        return acc, X_train, X_test_tensor.detach().numpy(), X_test_clean, y_train, y_test

    def load_dataset(self, dataset_fn=None, X=None, y=None):
        # load dataset
        if dataset_fn:
            X, y = dataset_fn(return_X_y=True)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_percentage, random_state=self.random_state)

        # keep test set before modifications
        X_test_clean = X_test.copy()

        return X_train, X_test, X_test_clean, y_train, y_test
