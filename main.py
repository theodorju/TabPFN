from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = TabPFNClassifier(device='cpu')
classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, y_test, return_winning_probability=True)