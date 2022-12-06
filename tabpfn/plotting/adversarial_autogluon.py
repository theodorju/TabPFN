import os
import shutil
import argparse
import pickle
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score
from sklearn.datasets import *


def run_autogluon_comparison(
        dataset_name="iris",
        lr=0.0025
):

    print_every = 4
    print("#" * 30)
    print(f"Dataset: {dataset_name} \nLearning rate: {lr}")

    results_file_path = f'../results/{dataset_name}_results_{lr}.pkl'

    # load results dictionary
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    # load necessary arrays
    X_train = np.load(f'../results/{dataset_name}_{lr}/X_train.npy')
    y_train = np.load(f'../results/{dataset_name}_{lr}/y_train.npy')
    X_test_clean = np.load(f'../results/{dataset_name}_{lr}/X_test_clean.npy')
    y_test = np.load(f'../results/{dataset_name}_{lr}/y_test.npy')

    for i, X_attacked in enumerate(results['tabPFN']['X_test']):

        if i == 0:

            # Fit autogluon
            print("Fitting Autogluon...")
            train_df = pd.DataFrame(np.concatenate(
                (X_train, y_train.astype(int).reshape(-1, 1)), axis=1),
                columns=[str(i) for i in range(X_train.shape[-1])] + ["label"])

            test_df = pd.DataFrame(X_test_clean, columns=[str(i) for i in range(X_test_clean.shape[-1])])

            agl_model = TabularPredictor(label='label')
            agl_model.fit(train_data=train_df, presets="best_quality")
            agl_preds = agl_model.predict(test_df)
            agl_preds = agl_preds.to_numpy()
            agl_acc = accuracy_score(y_test, agl_preds)
            print(f"\tAutoGluon initial accuracy: {agl_acc}")
            results['autogluon']['accuracy'].append(agl_acc)

        else:
            # predict Autogluon on attacked X
            test_modified_df = pd.DataFrame(X_attacked, columns=[str(i) for i in range(X_test_clean.shape[-1])])
            agl_preds_modified = agl_model.predict(test_modified_df)
            agl_preds_modified = agl_preds_modified.to_numpy()
            agl_acc = accuracy_score(y_test, agl_preds_modified)
            print(f"\tAutoGluon accuracy on {i * print_every} step: {agl_acc}")
            results['autogluon']['accuracy'].append(agl_acc)

    # Dump results back to file
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)

    # Cleanup
    if os.path.exists("AutogluonModels") and os.path.isdir("AutogluonModels"):
        shutil.rmtree("AutogluonModels")

    print("Exiting...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Adversarial comparisons",
        description="Run TabPFN adversarial attack and compare results to other frameworks"
    )

    parser.add_argument(
        "-o",
        "--openml",
        help="Run comparison on automl datasets",
        action="store_true",
    )

    args = parser.parse_args()
    is_openml = args.openml

    datasets_fn = [load_iris, load_breast_cancer, load_digits, None]
    lrs = [0.001, 0.0025, 0.005, 0.01, 0.1]

    if is_openml:

        openml_dataset_id = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37]

        for task_id in openml_dataset_id:
            print(f"Starting adversarial autogluon predictions for openml dataset: {task_id}...")
            for lr in lrs:
                try:
                    run_autogluon_comparison(dataset_name=str(task_id), lr=lr)
                except Exception as e:
                    print(f"Openml dataset {task_id} failed on autogluon with error:")
                    print(e)

    else:
        # Loop over datasets
        for dataset_fn in datasets_fn:

            dataset_name = "_".join(dataset_fn.__name__.split("_")[1:]) if dataset_fn is not None else "titanic"

            # Loop over learning rates
            for lr in lrs:

                # Run comparison
                run_autogluon_comparison(dataset_name=dataset_name, lr=lr)
