import torch
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tqdm import tqdm
import torch.optim as optim

# Load the dataset
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = TabPFNClassifier(device='cpu')
classifier.fit(X_train, y_train)

y_eval, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
before_score = accuracy_score(y_test, y_eval)
print('Before Accuracy', before_score)

"""
num_steps = 10
lmdb = 100
num_features = X.shape[-1]
rand_feature_indx = 2  # torch.randint(0, num_features,(1,)).item()
# X_attack_feature = X_test_[:, :, rand_feature_indx].detach().numpy().reshape(-1)

for step in tqdm(range(100)):
    optimizer.zero_grad()
    y_eval, p_eval, X_full_with_grad = classifier.predict(X_test_tensor.detach().numpy(), y_test, \
                                       return_winning_probability=True)
    # get new gradients
    # X_test_grad = X_full_with_grad.grad[X_train.shape[0]:, :].detach().squeeze().numpy()
    # update x_test
    # X_attack_feature += lmdb * X_test_grad[:, rand_feature_indx]
    optimizer.step()

    if step % 20 == 0:
        print("\n")
        # print(accuracy_score(y_test, y_eval))
        print("##"*20)

y_eval, p_eval, X_full_with_grad = classifier.predict(X_test_tensor, y_test, return_winning_probability=True)
y_eval_new, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
after_score = accuracy_score(y_test, y_eval_new)
print('After Accuracy', after_score)
"""