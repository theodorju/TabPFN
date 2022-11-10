from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tqdm import tqdm

class AdvAttackCompare:
    def __init__(self, dataset, classifier, step_size):
        self.dataset = dataset
        self.classifier = classifier
        self.step_size = step_size
    def attack_and_compare(self):
        self.load_and_initialize()
        lmdb = 100

        # num_features = X.shape[-1] ##TODO: wait for optimizer, number of features to ideally change
        rand_feature_indx = 2  # torch.randint(0, num_features,(1,)).item()
        X_attack_feature = self.X_test_[:, :, rand_feature_indx].detach().numpy().reshape(-1)

        for step in range(self.step_size):
            # print("Running Step NR {}".format(step+1))
            # Predict
            y_eval, p_eval, X_full_with_grad = self.classifier.predict(self.X_test, self.y_test, return_winning_probability=True)
            # get new gradients
            X_test_grad = X_full_with_grad.grad[self.X_train.shape[0]:, :].detach().squeeze().numpy()
            # update x_test
            X_attack_feature += lmdb * X_test_grad[:, rand_feature_indx] #TODO: what if multiple features were attacked? how to update and initialize?
            self.X_test[:, rand_feature_indx] = X_attack_feature

            print("X_test_grad", X_test_grad[45:, rand_feature_indx])
            print("X_test", self.X_test[45:, rand_feature_indx])
            print("##" * 20)

        y_eval_new, p_eval, X_full_with_grad = self.classifier.predict(self.X_test, self.y_test, return_winning_probability=True)
        after_score = accuracy_score(self.y_test, y_eval_new)
        print('After Accuracy', after_score)

    def load_and_initialize(self):
        self.X_train, self.X_test, y_train, self.y_test = train_test_split(self.dataset[0], self.dataset[1], test_size=0.33,
                                                            random_state=42)
        self.classifier.fit(self.X_train, y_train)
        y_eval, p_eval, X_full_with_grad = self.classifier.predict(self.X_test, self.y_test, return_winning_probability=True)
        self.before_score = accuracy_score(self.y_test, y_eval)
        print('Before Accuracy', self.before_score)
        self.X_test_ = X_full_with_grad[self.X_train.shape[0]:, :, :]


obj = AdvAttackCompare(load_iris(return_X_y=True), TabPFNClassifier(device='cpu'), 10)
obj.attack_and_compare()

