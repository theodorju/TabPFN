import numpy as np
import pickle
import torch
import torch.nn as nn

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier, check_is_fitted, check_array, \
    transformer_predict, get_params_from_config
from sklearn.metrics import accuracy_score

class AdversarialTabPFNClassifier(TabPFNClassifier):

    def __init__(self, device):
        self.device = device
        super(AdversarialTabPFNClassifier, self).__init__()

    def predict_proba_attack(
            self, X, y_test, optimizer, lr,
            num_steps=250,
            normalize_with_test=False,
            print_every=10,
            save_results=False,
            dataset_name=None
    ):
        """
        AutoML Lab Team Override
        """

        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X, force_all_finite=False)
        # Convert to tensors
        X_train_tensor = torch.from_numpy(self.X_).to(self.device).float()
        X_test_tensor = torch.from_numpy(X).to(self.device).float()

        # Activate gradient
        X_train_tensor.requires_grad = True
        X_test_tensor.requires_grad = True

        # Results dictionary
        results = {
            "tabPFN": {
                "loss": [],
                "accuracy": [],
                "X_test": [],
                "l2_norm": [],
                "l2_norm_overall": [],
                "learning_rate": lr,
                "dataset_name": dataset_name,
            },
            "askl2": {
                "accuracy": [],
                "failed": False,
            },
            "autogluon": {
                "accuracy": [],
                "failed": False,
            },
            "xgboost": {
                "accuracy": [],
                "failed": False,
            },
            "mlp": {
                "accuracy": [],
                "failed": False,
            }
        }

        # Instantiate optimizer
        optim = optimizer([X_test_tensor], lr=lr, maximize=True)
        print('Applying adversarial attack on {}:'
              '\n \t Optimizer: {}'
              '\n \t Learning Rate: {}'
              '\n \t Number of steps: {}'.format(dataset_name, type(optim).__name__, lr, num_steps))

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Concatenate
        X_full = torch.concat([X_train_tensor, X_test_tensor], axis=0).float().unsqueeze(1)
        y_full = np.concatenate([self.y_, np.zeros_like(X[:, 0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)
        eval_pos = self.X_.shape[0]

        # initialize to calculate l2-norm on the fly
        previous_X_test = X

        for step in range(0, num_steps + 1):

            optim.zero_grad()

            prediction = transformer_predict(self.model[2], X_full, y_full, eval_pos,
                                             device=self.device,
                                             style=self.style,
                                             inference_mode=False,
                                             preprocess_transform='none',  # if self.no_preprocess_mode else 'mix',
                                             normalize_with_test=normalize_with_test,
                                             N_ensemble_configurations=self.N_ensemble_configurations,
                                             softmax_temperature=self.temperature,
                                             combine_preprocessing=self.combine_preprocessing,
                                             multiclass_decoder=self.multiclass_decoder,
                                             feature_shift_decoder=self.feature_shift_decoder,
                                             differentiable_hps_as_style=self.differentiable_hps_as_style,
                                             **get_params_from_config(self.c),
                                             )

            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            pred = prediction.squeeze()
            loss = loss_fn(pred, y_test_tensor)
            acc = accuracy_score(np.argmax(pred.detach().numpy(), axis=-1), y_test_tensor.detach().numpy())

            loss.backward()
            optim.step()

            # print every print_every steps and on the final step
            if (step % print_every == 0) or (step == num_steps):
                print(f"Step: {step}"
                      f"\n \t Loss: {loss.item():.5f}" 
                      f"\n \t Accuracy: {acc:.5f}")

                if save_results:
                    results["tabPFN"]["loss"].append(loss.item())
                    results["tabPFN"]["accuracy"].append(acc)
                    results["tabPFN"]["X_test"].append(X_test_tensor.detach().numpy().copy())
                    results["tabPFN"]["l2_norm"].append(
                        np.linalg.norm(X_test_tensor.detach().numpy() - previous_X_test, ord=2))
                    results["tabPFN"]["l2_norm_overall"].append(
                        np.linalg.norm(X_test_tensor.detach().numpy() - X, ord=2))

            previous_X_test = X_test_tensor.detach().numpy().copy()
            X_full = torch.concat([X_train_tensor, X_test_tensor], axis=0).float().unsqueeze(1)

            prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        if save_results:
            with open(f'../results/{dataset_name}_results_{lr}.pkl', 'wb') as f:
                pickle.dump(results, f)

        return prediction_.detach().cpu().numpy(), X_full, X_test_tensor

    def predict_attack(self, X, y_test, optimizer, lr, num_steps=250, return_winning_probability=False,
                       normalize_with_test=False, print_every=10, save_results=False, dataset_name=None):
        """
        AutoML Lab Team Override
        """

        # Perform adversarial attack and predict
        p, x_full, x_test = self.predict_proba_attack(X, y_test,
                                                      optimizer=optimizer,
                                                      lr=lr,
                                                      num_steps=num_steps,
                                                      normalize_with_test=normalize_with_test,
                                                      print_every=print_every,
                                                      save_results=save_results,
                                                      dataset_name=dataset_name)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))

        return (y, p.max(axis=-1), x_full, x_test) if return_winning_probability else (y, x_full, x_test)
