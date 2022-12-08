import os
# ASKL2 failed on digits with an openblas error locally. this fixed it.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch.optim as optim
import pickle
import numpy as np
import xgboost
import openml
import argparse

from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import *
from sklearn.neural_network import MLPClassifier
from tabpfn.adversarial.adversarial import AdversarialTabPFNInterface


def run_comparison(
        lr=0.0025,
        num_steps=4,
        optim=optim.Adam,
        dataset_fn=None,
        X=None,
        y=None,
        dataset_name='titanic',
        models='all',
        test_percentage=0.2
) -> None:
    """
    Run adversarial comparison on a dataset. This function is used to run the adversarial comparisons. It performs
    the adversarial attack on TabPFN using an optimizer and learning rate for some number of steps.
    It then performs predictions using the attacked dataset on the rest of the baselines.

    Baselines include:
        - MLP: The sklearn MLP classifier with default parameters.
        - XGBoost: XGBoost classifier with default parameters.
        - AutoSklearn: AutoML framework with default parameters.

    The comparisons on AutoGluon are performed in a different script since currently execution of AutoGluon on the
    same environment that TabPFN is installed is not supported.

    Args:
        lr (float): The learning rate of the optimizer performing the gradient ascent (adversarial attack).
            Default: 0.0025
        num_steps (int): Number of steps of the adversarial attack. Default: 100
        optim (torch.nn.optim): The optimizer to use for the adversarial attack. Default: torch.optim.Adam
        dataset_fn: The function to use to load the dataset. Default: None
        X (ndarray): The dataset to use. Default: None (Either dataset_fn or X, y must be provided)
        y (ndarray): The labels of the dataset. Default: None (Either dataset_fn or X, y must be provided)
        dataset_name (str): The name of the dataset used for logging. Default: titanic
        models (str): The models to use for the comparison. Default: "all" (All models are used)
        test_percentage (float): The percentage of the dataset to use for testing. Default: 0.2

    Returns:
        None
    """

    # If dataset_fn is not provided use the specified dataset name, otherwise parse the dataset function and use that
    dataset_name = dataset_name if dataset_fn is None else "_".join(dataset_fn.__name__.split("_")[1:])
    print_every = 4

    # If neither dataset nor X, y are provided load the titanic dataset
    if dataset_fn is None and (X is None or y is None):
        X = np.load(f'../adversarial_xy_data/{dataset_name}/X.npy')
        y = np.load(f'../adversarial_xy_data/{dataset_name}/y.npy')

    # Setup directory
    if not os.path.exists('../results/'):
        os.mkdir('../results/')

    # Instantiate the adversarial attack interface
    adv = AdversarialTabPFNInterface(dataset_fn=dataset_fn,
                                     optimizer=optim,
                                     num_steps=num_steps,
                                     lr=lr,
                                     save_results=True,
                                     X_full=X,
                                     y_full=y,
                                     dataset_name=dataset_name,
                                     test_percentage=test_percentage,
                                     print_every=print_every
                                     )

    # Perform attack on TabPFN, return train and clean test set, modified test set, train and test labels
    # the train test split is performed internally in the adversarial_attack() method
    X_train, X_test, X_test_clean, y_train, y_test = adv.adversarial_attack()

    # Save splits for autogluon
    print("Saving numpy arrays for AutoGluon...")
    if not os.path.exists(f"../results/{dataset_name}_{lr}"):
        os.mkdir(f"../results/{dataset_name}_{lr}")

    # Save the splits for AutoGluon
    np.save(f"../results/{dataset_name}_{lr}/X_train", X_train)
    np.save(f"../results/{dataset_name}_{lr}/X_test_clean", X_test_clean)
    np.save(f"../results/{dataset_name}_{lr}/y_train", y_train)
    np.save(f"../results/{dataset_name}_{lr}/y_test", y_test)

    # Load results file, generated automatically during TabPFN adversarial attack
    results_file_path = f'../results/{dataset_name}_results_{lr}.pkl'

    # Load the results file
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    # Loop over the modified test sets generated during TabPFN adversarial attacks and perform comparisons
    # We assume that if the attack fails at some step, the results file will not contain the results for that and
    # any upcoming steps
    for i, X_attacked in enumerate(results['tabPFN']['X_test']):

        # For the first iteration fit models and predict on the unmodified test set
        if i == 0:

            ################### Autosklearn2 fitting ###################
            if models in ("askl", "all"):
                try:
                    # Fit Autosklearn
                    askl2_model = AutoSklearnClassifier(time_left_for_this_task=120)
                    print("Fitting Autosklearn2...")
                    askl2_model.fit(X_train, y_train)
                    predictions = askl2_model.predict(X_test_clean)
                    askl2_acc = accuracy_score(y_test, predictions)
                    print(f"\tAutosklearn initial accuracy: {askl2_acc}")
                    results['askl2']['accuracy'].append(askl2_acc)

                except Exception:
                    results['askl2']['failed'] = True
                    print(f"askl2 failed to fit on {dataset_name} for learning rate: {lr}")

            ################### XGBoost fitting ###################
            try:
                if models in ("xgb", "all"):
                    # Fit XGBoost
                    print("Fitting XGBoost...")
                    xgb_model = xgboost.XGBClassifier()
                    xgb_model.fit(X_train, y_train)
                    xgb_preds = xgb_model.predict(X_test_clean)
                    xgb_acc = accuracy_score(y_test, xgb_preds)
                    print(f"\tXGBoost initial accuracy: {xgb_acc}")
                    results['xgboost']['accuracy'].append(xgb_acc)

            except Exception:
                results['xgboost']['failed'] = True
                print(f"XGBoost failed to fit on {dataset_name} for learning rate: {lr}")

            ################### MLP fitting ###################
            try:
                if models in ("mlp", "all"):
                    print("Fitting SKLearn MLP...")
                    mlp_model = MLPClassifier(max_iter=200)
                    mlp_model.fit(X_train, y_train)
                    mlp_preds = mlp_model.predict(X_test_clean)
                    mlp_acc = accuracy_score(y_test, mlp_preds)
                    print(f"\tSklearn MLP initial accuracy: {mlp_acc}")
                    results["mlp"]['accuracy'].append(mlp_acc)

            except Exception:
                results['mlp']['failed'] = True
                print(f"MLP failed to fit on {dataset_name} for learning rate: {lr}")

        # For all subsequent iterations, use the already fitted models and only perform predictions in the modified
        # test sets.
        # Note that the test have been modified by adversarial attacks (gradient ascent) on TabPFN, not on each
        # individual model used here only for predictions.
        else:
            try:
                ################### Autosklearn2 prediction ###################
                if models in ("askl", "all") and not results['askl2']['failed']:
                    # predict Autosklearn on attacked X
                    askl2_preds_modified = askl2_model.predict(X_attacked)
                    askl2_acc = accuracy_score(y_test, askl2_preds_modified)
                    print(f"\tAutosklearn accuracy on {i * print_every} step: {askl2_acc}")
                    results['askl2']['accuracy'].append(askl2_acc)
            except Exception:
                results['askl2']['failed'] = True
                print(f" askl2 failed to predict on {dataset_name} for learning rate: {lr} on step: {i}")

            ################### XGBoost prediction ###################
            try:
                if models in ("xgb", "all") and not results['xgboost']['failed']:
                    # predict XGBoost on attacked X
                    xgb_preds_modified = xgb_model.predict(X_attacked)
                    xgb_acc = accuracy_score(y_test, xgb_preds_modified)
                    print(f"\tXGBoost accuracy on {i * print_every} step: {xgb_acc}")
                    results['xgboost']['accuracy'].append(xgb_acc)
            except Exception:
                results['xgboost']['failed'] = True
                print(f" XGBoost failed to predict on {dataset_name} for learning rate: {lr} on step: {i}")
            ################### MLP prediction ###################
            try:
                if models in ("mlp", "all") and not results['mlp']['failed']:
                    mlp_preds_modified = mlp_model.predict(X_attacked)
                    mlp_acc = accuracy_score(y_test, mlp_preds_modified)
                    print(f"\tSklearn MLP accuracy on {i * print_every} step: {mlp_acc}")
                    results['mlp']['accuracy'].append(mlp_acc)
            except Exception:
                results['mlp']['failed'] = True
                print(f" MLP failed to predict on {dataset_name} for learning rate: {lr} on step: {i}")

    # save results
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)


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

    parser.add_argument(
        "-s",
        "--number_of_steps",
        help="Number of steps to run the attack for",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-d",
        "--openml_datasets",
        nargs="*",  # 0 or more values expected => creates a list
        help="OpenML dataset ids to run the attack on",
        type=int,
        default=[11, 14, 15, 16, 18, 22, 23, 29, 31, 37],
    )

    parser.add_argument(
        "-lr",
        "--learning_rates",
        nargs="*",  # 0 or more values expected => creates a list
        help="Learning rates to run the attack on",
        type=float,
        default=[0.001, 0.0025, 0.005, 0.01, 0.1]
    )

    parser.add_argument(
        "-t",
        "--test_percentage",
        help="Percentage of the dataset to use as test set",
        type=float,
        default=0.2,
    )

    # Parse arguments
    args = parser.parse_args()
    is_openml = args.openml

    # Number of adversarial attack steps
    num_steps = args.number_of_steps

    # learning rates
    lrs = args.learning_rates

    # TabPFN current limit is 1000 examples
    tabpfn_limit = 1000

    if is_openml:

        openml_dataset_id = args.openml_datasets

        for task_id in openml_dataset_id:
            try:
                print(f"Starting adversarial comparisons for openml dataset: {task_id}...")

                task = openml.tasks.get_task(task_id, download_data=True)
                dataset = task.get_dataset()

                X, y, categorical_indicator, _ = dataset.get_data(
                    dataset_format='array',
                    target=dataset.default_target_attribute
                )

                test_percentage = args.test_percentage
                num_examples = X.shape[0]

                # Skip datasets with more than 2000 features
                if num_examples > 2000:
                    continue

                if num_examples > tabpfn_limit:
                    test_percentage = np.round(1 - tabpfn_limit / num_examples, 2)

                for lr in lrs:
                    run_comparison(lr,
                                   num_steps=num_steps,
                                   X=X, y=y,
                                   dataset_name=task_id,
                                   models="all",
                                   test_percentage=test_percentage
                                   )

            except Exception as e:
                print(f"Openml dataset {task_id} failed with error:")
                print(e)

    else:
        # Setup
        datasets_fn = [load_iris, load_breast_cancer, load_digits, None]

        # Digits requires different test percentage to result in 1000 training examples due to TabPFN restrictions
        test_percentage = [0.2, 0.2, 0.4435, 0.2, 0.2]

        # Loop over datasets
        for i, dataset_fn in enumerate(datasets_fn):

            # Loop over learning rates
            for lr in lrs:

                # Call for specific dataset
                if dataset_fn is None:
                    run_comparison(lr=lr,
                                   dataset_fn=dataset_fn,
                                   models='all',
                                   test_percentage=test_percentage[i],
                                   num_steps=num_steps,
                                   dataset_name='titanic'
                                   )

                # Call for sklearn datasets
                else:
                    # Run comparison
                    run_comparison(lr=lr,
                                   dataset_fn=dataset_fn,
                                   models='all',
                                   test_percentage=test_percentage[i],
                                   num_steps=num_steps
                                   )
