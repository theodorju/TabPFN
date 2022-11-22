from tabpfn.adversarial import *
from sklearn.datasets import *
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import autosklearn.classification
from autogluon.tabular import TabularDataset, TabularPredictor

optim = optim.Adam
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60)
datasets_fn = [load_iris]
all_accs = []

for dataset_fn in datasets_fn:

    dataset_name = "_".join(dataset_fn.__name__.split("_")[1:])

    # TODO: Fix keys in code
    results = {
        "accuracy": {'tabPFN': [], 'askl': [], 'gluon': []},
        'learning_rate': 0.025,
        'dataset_name': dataset_name
    }


    for i, num_step in enumerate(range(0, 20)):

        # PFN with Adversarial attack
        adv = AdversarialTabPFN(datasets_fn=dataset_fn, optimizer=optim, num_steps=num_step, lr=0.025)
        acc, X_train, X_test, X_test_clean, y_train, y_test = adv.adversarial_attack()
        accs['tabPFN'].append(acc)

        # AutoSklearn with Adversarial attack
        if i == 0:
            cls.fit(X_train, y_train)
            predictions = cls.predict(X_test_clean)
            acc = accuracy_score(y_test, predictions)
            accs['askl'].append(acc)
            print('AutoSklearn Fitted on Training with accuracy: {}'.format(acc))
        else:
            predictions_modified = cls.predict(X_test)
            acc_modified = accuracy_score(y_test, predictions_modified)
            accs['askl'].append(acc_modified)

        # Autogluon with Adversarial attack
        if i == 0:
            train_df = pd.DataFrame(np.concatenate((X_train, y_train.astype(int).reshape(-1, 1)), axis=1),
                                    columns=[str(i) for i in range(X_train.shape[-1])] + ["label"])

            test_df = pd.DataFrame(X_test_clean, columns=[str(i) for i in range(X_test_clean.shape[-1])])

            predictor = TabularPredictor(label='label').fit(train_data=train_df, presets="best_quality")
            predictions = predictor.predict(test_df)
            y_pred = predictions.to_numpy()
            accs['gluon'].append(accuracy_score(y_test, y_pred))

        else:

            test_modified_df = pd.DataFrame(X_test, columns=[str(i) for i in range(X_test.shape[-1])])
            predictions = predictor.predict(test_modified_df)
            y_pred = predictions.to_numpy()
            accs['gluon'].append(accuracy_score(y_test, y_pred))

    with open(f'plots/tabpfn_{dataset_name}_acc_vs_num_steps.pkl', 'wb') as f:
        pickle.dump(accs, f)
    all_accs.append(accs)
