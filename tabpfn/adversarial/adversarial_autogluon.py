import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score
from sklearn.datasets import *

datasets_fn = [load_breast_cancer]
base_dir = "data/"
for dataset_fn in datasets_fn:
    dataset_name = "_".join(dataset_fn.__name__.split("_")[1:])

    train_data = TabularDataset(base_dir + dataset_name + '_train.csv')
    test_data = TabularDataset(base_dir + dataset_name + '_test.csv')
    predictor = TabularPredictor(label='label').fit(train_data=train_data, presets="best_quality")

    y_test = np.load(base_dir + dataset_name + "_y_test.npy")
    predictions = predictor.predict(test_data)
    y_pred = predictions.to_numpy()
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    test_data_modified = TabularDataset(base_dir + dataset_name + '_test_modified.csv')
    predictions = predictor.predict(test_data_modified)
    y_pred_modified = predictions.to_numpy()
    acc_modified = accuracy_score(y_test, y_pred_modified)
    print(acc_modified)
