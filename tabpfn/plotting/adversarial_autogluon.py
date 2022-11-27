import os
import shutil
import pickle
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score


def run_autogluon_comparison(dataset_name="iris", lr=0.0025):

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

    # Loop over datasets
    for dataset_name in ["breast_cancer"]:

        # Loop over learning rates
        for lr in [0.01]:

            # Run comparison
            run_autogluon_comparison(dataset_name=dataset_name, lr=lr)
