import torch.optim as optim
import numpy as np
import os
import pandas as pd
import pickle
import autosklearn.classification
import xgboost

from sklearn.metrics import accuracy_score
from sklearn.datasets import *
from sklearn.neural_network import MLPClassifier
from autogluon.tabular import TabularPredictor
from tabpfn.adversarial.adversarial import AdversarialTabPFN

optim = optim.Adam
datasets_fn = [load_iris]
lr = 0.0025

def run_comparisson(dataset_fn=None, X=None, y=None, dataset_name=None, models=None):

    # Setup adversarial attack
    adv = AdversarialTabPFN(dataset_fn=dataset_fn,
                            optimizer=optim,
                            num_steps=24,
                            lr=lr,
                            save_results=True,
                            X_full=X,
                            y_full=y,
                            dataset_name=dataset_name)

    # Perform attack on TabPFN, return train and clean test set, modified test set, train and test labels
    # the train test split is performed internally in the adversarial_attack() method
    X_train, X_test, X_test_clean, y_train, y_test = adv.adversarial_attack()

    # Load results file, generated automatically during TabPFN adversarial attack
    results_file_path = f'../results/{dataset_name}_results_{lr}.pkl'

    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    # Loop over the modified test sets generated during TabPFN adversarial attacks and perform comparisons
    for i, X_attacked in enumerate(results['tabPFN']['X_test']):

        # For the first iteration fit models and predict on the unmodified test set
        if i == 0:

            if models in ("askl", "all"):
                # Fit Autosklearn
                askl2_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60)
                print("Fitting Autosklearn2...")
                askl2_model.fit(X_train, y_train)
                predictions = askl2_model.predict(X_test_clean)
                askl2_acc = accuracy_score(y_test, predictions)
                results['askl2']['accuracy'].append(askl2_acc)

            if models in ("agl", "all"):
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
                results['autogluon']['accuracy'].append(agl_acc)

            if models in ("xgb", "all"):
                # Fit XGBoost
                print("Fitting XGBoost...")
                xgb_model = xgboost.XGBClassifier()
                xgb_model.fit(X_train, y_train)
                xgb_preds = xgb_model.predict(X_test_clean)
                xgb_acc = accuracy_score(y_test, xgb_preds)
                results['xgboost']['accuracy'].append(xgb_acc)

            if models in ("mlp", "all"):
                print("Fitting SKLearn MLP...")
                mlp_model = MLPClassifier()
                mlp_model.fit(X_train, y_train)
                mlp_preds = mlp_model.predict(X_test_clean)
                mlp_acc = accuracy_score(y_test, mlp_preds)
                results["mlp"]['accuracy'].append(mlp_acc)

        # for all subsequent iterations, use the already fitted models and only perform predictions in the modified
        # test sets.
        # Note that the test have been modified by adversarial attacks (gradient ascent) on TabPFN, not on each
        # individual model used here only for predictions.
        else:

            if models in ("askl", "all"):
                # predict Autosklearn on attacked X
                askl2_preds_modified = askl2_model.predict(X_attacked)
                askl2_acc = accuracy_score(y_test, askl2_preds_modified)
                results['askl2']['accuracy'].append(askl2_acc)

            if models in ("agl", "all"):
                # predict Autogluon on attacked X
                test_modified_df = pd.DataFrame(X_attacked, columns=[str(i) for i in range(X_test.shape[-1])])
                agl_preds_modified = agl_model.predict(test_modified_df)
                agl_preds_modified = agl_preds_modified.to_numpy()
                agl_acc = accuracy_score(y_test, agl_preds_modified)
                results['autogluon']['accuracy'].append(agl_acc)

            if models in ("xgb", "all"):
                # predict XGBoost on attacked X
                xgb_preds_modified = xgb_model.predict(X_attacked)
                xgb_acc = accuracy_score(y_test, xgb_preds_modified)
                results['xgboost']['accuracy'].append(xgb_acc)

            if models in ("mlp", "all"):
                print(X_attacked[1, :5])
                mlp_preds_modified = mlp_model.predict(X_attacked)
                mlp_acc = accuracy_score(y_test, mlp_preds_modified)
                results['mlp']['accuracy'].append(mlp_acc)

    # Cleanup autogluon
    if os.path.exists("AutogluonModels") and os.path.isdir("AutogluonModels"):
        shutil.rmtree("AutogluonModels")

    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    run_comparisson(dataset_fn=load_breast_cancer, models='all')

