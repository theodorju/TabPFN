import os
# ASKL2 failed on digits with an openblas error locally. this fixed it.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch.optim as optim
import pickle
import numpy as np
import xgboost

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
        dataset_name=None,
        models=None,
        test_percentage=0.2
):

    dataset_name = dataset_name if dataset_fn is None else "_".join(dataset_fn.__name__.split("_")[1:])
    print_every = 4

    if dataset_fn is None:
        X = np.load(f'../adversarial_xy_data/{dataset_name}/X.npy')
        y = np.load(f'../adversarial_xy_data/{dataset_name}/y.npy')

    # Setup directory
    if not os.path.exists('../results/'):
        os.mkdir('../results/')

    # Setup adversarial attack
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

    np.save(f"../results/{dataset_name}_{lr}/X_train", X_train)
    np.save(f"../results/{dataset_name}_{lr}/X_test_clean", X_test_clean)
    np.save(f"../results/{dataset_name}_{lr}/y_train", y_train)
    np.save(f"../results/{dataset_name}_{lr}/y_test", y_test)

    # Load results file, generated automatically during TabPFN adversarial attack
    results_file_path = f'../results/{dataset_name}_results_{lr}.pkl'

    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    # Loop over the modified test sets generated during TabPFN adversarial attacks and perform comparisons
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

            ################### Autosklearn2 prediction ###################
            if models in ("askl", "all") and not results['askl2']['failed']:
                # predict Autosklearn on attacked X
                askl2_preds_modified = askl2_model.predict(X_attacked)
                askl2_acc = accuracy_score(y_test, askl2_preds_modified)
                print(f"\tAutosklearn accuracy on {i * print_every} step: {askl2_acc}")
                results['askl2']['accuracy'].append(askl2_acc)


            ################### XGBoost prediction ###################
            if models in ("xgb", "all") and not results['xgboost']['failed']:
                # predict XGBoost on attacked X
                xgb_preds_modified = xgb_model.predict(X_attacked)
                xgb_acc = accuracy_score(y_test, xgb_preds_modified)
                print(f"\tXGBoost accuracy on {i * print_every} step: {xgb_acc}")
                results['xgboost']['accuracy'].append(xgb_acc)

            ################### MLP prediction ###################
            if models in ("mlp", "all") and not results['mlp']['failed']:
                mlp_preds_modified = mlp_model.predict(X_attacked)
                mlp_acc = accuracy_score(y_test, mlp_preds_modified)
                print(f"\tSklearn MLP accuracy on {i * print_every} step: {mlp_acc}")
                results['mlp']['accuracy'].append(mlp_acc)

    # save results
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    # Number of adversarial attack steps
    num_steps = 100

    # Setup
    datasets_fn = [load_iris, load_breast_cancer, load_digits, None]

    # Digits requires different test percentage to result in 1000 training examples due to TabPFN restrictions
    test_percentage = [0.2, 0.2, 0.4435, 0.2]
    lrs = [0.001, 0.0025, 0.005, 0.01, 0.1]

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
