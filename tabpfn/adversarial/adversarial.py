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
            lr=0.005,
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
        self.optimizer = optimizer
        self.num_steps = num_steps

        if datasets_fn is None and (X_full is None and y_full is None):
            print("No dataset or X, y provided. Using Breast Cancer Dataset from sklearn.")
            self.dataset_fn = load_breast_cancer

    def adversarial_attack(self):
        """
        This function performs the following steps:
            1. Load the dataset based on info provided (either sklearn dataset or X, y ndarrays).
            2. Fit the TabPFN classifier and print the initial prediction on the dataset.
            3. Perform a number of adversarial gradient ascent steps.
            4. Print the final accuracy on the adversarial test dataset.

        """

        # Create X, y train/test arrays
        X_train, X_test, y_train, y_test = \
            self.load_dataset(self.dataset_fn, self.X_full, self.y_full)

        # Keep test set before modifications
        X_test_clean = X_test.copy()

        # Instantiate the classifier
        classifier = TabPFNClassifier(device='cpu')

        # Fit classifier
        classifier.fit(X_train, y_train)

        # Evaluate the classifier, this calls the default predict method, no attacks or gradients so far
        y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

        # Get the accuracy before
        before_acc = accuracy_score(y_test, y_eval)
        print(f"Initial accuracy: {before_acc}")

        # Perform adversarial attack for specified number of steps, this calls our new adversarial function
        y_eval_before, p_eval_before, X_full_with_grad, X_test_tensor = \
            classifier.predict_attack(X_test, y_test,
                                      optimizer=self.optimizer,
                                      lr=self.lr,
                                      num_steps=self.num_steps,
                                      return_winning_probability=True
                                      )

        # Print accuracy after all attack steps
        acc = accuracy_score(y_test, y_eval_before)
        print(f"Accuracy: {acc}")

        return acc, X_train, X_test_tensor.detach().numpy(), X_test_clean, y_train, y_test

    def load_dataset(self, dataset_fn=None, X=None, y=None):
        # If a sklearn type dataset is specified
        if dataset_fn:
            # Populate X, y
            X, y = dataset_fn(return_X_y=True)

        # Split the dataset
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_percentage, random_state=self.random_state)

        return X_train, X_test, y_train, y_test
