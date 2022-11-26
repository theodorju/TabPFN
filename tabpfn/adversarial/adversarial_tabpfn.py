import os
import numpy as np
import pandas as pd
import argparse
from tabpfn.adversarial.adversarial import *
from sklearn.datasets import *


def adversarial_sklearn_datasets(dataset_fn=None, X=None, y=None, lr=0.001, num_steps=20):

    adv = AdversarialTabPFN(dataset_fn=dataset_fn, num_steps=8, lr=lr,
                            print_every=4, save_results=True)
    X_train, X_test, X_test_clean, y_train, y_test = adv.adversarial_attack()

    # path_exists = os.path.exists("data")
    # if not path_exists:
    #     # Create path if it does not exist
    #     os.makedirs("data")
    # basedir = "data/"
    # np.save(basedir + dataset_name + "_X_train.npy", X_train)
    # np.save(basedir + dataset_name + "_y_train.npy", y_train.astype(int))
    # np.save(basedir + dataset_name + "_X_test_modified.npy", X_test)
    # np.save(basedir + dataset_name + "_X_test.npy", X_test_clean)
    # np.save(basedir + dataset_name + "_y_test.npy", y_test.astype(int))
    #
    # train_df = pd.DataFrame(
    #     np.concatenate((X_train, y_train.astype(int).reshape(-1, 1)), axis=1),
    #     columns=[str(i) for i in range(X_train.shape[-1])] + ["label"])
    #
    # train_df.to_csv(basedir + dataset_name + "_train.csv", index=False)
    #
    # test_df = pd.DataFrame(X_test_clean, columns=[str(i) for i in range(X_test_clean.shape[-1])])
    # test_df.to_csv(basedir + dataset_name + "_test.csv", index=False)
    #
    # test_modified_df = pd.DataFrame(X_test, columns=[str(i) for i in range(X_test.shape[-1])])
    # test_modified_df.to_csv(basedir + dataset_name + "_test_modified.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=20
    )

    args = parser.parse_args()
    lr, num_steps = args.learning_rate, args.steps

    ds = load_breast_cancer
    adversarial_sklearn_datasets(ds, lr, num_steps)

