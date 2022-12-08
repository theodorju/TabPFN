"""Perform a single prediction using adversarial dataset on AutoSklearn classifier."""
import autosklearn.classification
import numpy as np
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60)

datasets_fn = [load_breast_cancer]
base_dir = 'data/'

for dataset_fn in datasets_fn:
    dataset_name = "_".join(dataset_fn.__name__.split("_")[1:])

    X_train = np.load(base_dir + dataset_name + "_X_train.npy")
    X_test = np.load(base_dir + dataset_name + "_X_test.npy")
    X_test_modified = np.load(base_dir + dataset_name + "_X_test_modified.npy")
    y_train = np.load(base_dir + dataset_name + "_y_train.npy").astype(int)
    y_test = np.load(base_dir + dataset_name + "_y_test.npy").astype(int)

    print("Fitting")
    cls.fit(X_train, y_train)

    predictions = cls.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy on clean dataset: {acc}")

    predictions_modified = cls.predict(X_test_modified)
    acc_modified = accuracy_score(y_test, predictions_modified)
    print(f"Accuracy on modified dataset: {acc_modified}")
