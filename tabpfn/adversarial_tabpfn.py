from tabpfn.adversarial import *
from sklearn.datasets import *
import torch.optim as optim
import numpy as np
import pandas as pd

optim = optim.Adam

datasets_fn = [load_breast_cancer]

for dataset_fn in datasets_fn:

    dataset_name = "_".join(dataset_fn.__name__.split("_")[1:])

    adv = AdversarialTabPFN(datasets_fn=dataset_fn, optimizer=optim, num_steps=2, lr=0.01)
    acc, X_train, X_test, X_test_clean, y_train, y_test = adv.adversarial_attack()

    np.save(dataset_name + "_X_train.npy", X_train)
    np.save(dataset_name + "_y_train.npy", y_train.astype(int))
    np.save(dataset_name + "_X_test_modified.npy", X_test)
    np.save(dataset_name + "_X_test.npy", X_test_clean)
    np.save(dataset_name + "_y_test.npy", y_test.astype(int))

    train_df = pd.DataFrame(np.concatenate((X_train, y_train.astype(int).reshape(-1, 1)), axis=1),
                            columns=[str(i) for i in range(X_train.shape[-1])] + ["label"])
    train_df.to_csv(dataset_name + "_train.csv", index=False)

    test_df = pd.DataFrame(X_test_clean, columns=[str(i) for i in range(X_test_clean.shape[-1])])
    test_df.to_csv(dataset_name + "_test.csv", index=False)

    test_modified_df = pd.DataFrame(X_test, columns=[str(i) for i in range(X_test.shape[-1])])
    test_modified_df.to_csv(dataset_name + "_test_modified.csv", index=False)
