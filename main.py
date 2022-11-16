import torch
from sklearn.metrics import accuracy_score
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tqdm import tqdm


class Adversarial_TabPFN:
    def __init__(
            self,
            datasets_fn=None,
            lmbd=0.1,
            random_state=42,
            test_size=0.45,
            num_attack_steps=100
    ):

        self.datasets_fn = datasets_fn if datasets_fn is not None else [load_breast_cancer]
        self.lmbd = lmbd
        self.random_state = random_state
        self.test_size = test_size
        self.num_attack_steps = num_attack_steps

    def adversarial_attack(self):

        for dataset_fn in self.datasets_fn:
            print(f"Currently running for: {' '.join(dataset_fn.__name__.split('_')[1:])}")

            X_train, X_test, X_test_clean, y_train, y_test = self.load_and_initialize(dataset_fn)

            num_train_examples, num_features = X_train.shape
            num_test, _ = X_test.shape

            classifier = TabPFNClassifier(device='cpu')
            classifier.fit(X_train, y_train)

            y_eval_before, p_eval_before, X_full_with_grad = \
                classifier.predict(X_test, y_test, return_winning_probability=True)

            # Get the score before performing adversarial attack
            before_score = accuracy_score(y_test, y_eval_before)
            print(f"Initial accuracy: {before_score}")

            # Get the full grad
            grad = X_full_with_grad.grad[num_train_examples:, :, :].detach().numpy().squeeze()

            X_test = X_test + self.lmbd * grad

            for step in range(1, self.num_attack_steps + 1):
                y_eval, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)

                grad = X_full_with_grad.grad[num_train_examples:, :, :].detach().numpy().squeeze()

                X_test = X_test + self.lmbd * grad

                if step % 10 == 0:
                    acc = accuracy_score(y_test, y_eval)
                    print(f"accuracy after {step} steps: {acc}")

            y_eval_after, p_eval_after, X_full_with_grad = \
                classifier.predict(X_test, y_test, return_winning_probability=True)

            view_before = p_eval_before.reshape(-1, 1)
            view_after = p_eval_after.reshape(-1, 1)

            after_score = accuracy_score(y_test, y_eval_after)

            print(f"Final accuracy: {after_score}")

    def load_and_initialize(self, dataset_fn):
        # load dataset
        X, y = dataset_fn(return_X_y=True)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # keep test set before modifications
        X_test_clean = X_test.copy()

        return X_train, X_test, X_test_clean, y_train, y_test


datasets_fn = [load_iris, load_breast_cancer, load_wine]
obj = Adversarial_TabPFN(datasets_fn=datasets_fn)
obj.adversarial_attack()

# load_fns = [load_iris, load_breast_cancer, load_digits, load_wine]
#
# failing_fns = [load_diabetes]  # considers it classification even though it's regression
