from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tqdm import tqdm
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = TabPFNClassifier(device='cpu')
classifier.fit(X_train, y_train)
y_eval, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
before_score = accuracy_score(y_test, y_eval)
print('Before Accuracy', before_score)

X_test_ = X_full_with_grad[X_train.shape[0]:, :, :]
X_test_grad = X_full_with_grad.grad[X_train.shape[0]:, :, :]
num_steps = 10
lmdb = 100
num_features = X.shape[-1]
rand_feature_indx = 2  # torch.randint(0, num_features,(1,)).item()
X_attack_feature = X_test_[:, :, rand_feature_indx].detach().numpy().reshape(-1)
for step in range(num_steps):
    #print("Running Step NR {}".format(step+1))
    # Predict
    y_eval, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
    # get new gradients
    X_test_grad = X_full_with_grad.grad[X_train.shape[0]:, :].detach().squeeze().numpy()
    # update x_test
    X_attack_feature += lmdb * X_test_grad[:, rand_feature_indx]
    X_test[:, rand_feature_indx] = X_attack_feature

    print("X_test_grad", X_test_grad[45:, rand_feature_indx])
    print("X_test", X_test[45:, rand_feature_indx])
    print("##"*20)

y_eval, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
y_eval_new, p_eval, X_full_with_grad = classifier.predict(X_test, y_test, return_winning_probability=True)
after_score = accuracy_score(y_test, y_eval_new)
print('After Accuracy', after_score)
