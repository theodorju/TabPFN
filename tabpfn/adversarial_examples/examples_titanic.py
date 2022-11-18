import os
import numpy as np
import pandas as pd
from tabpfn.adversarial import Adversarial_TabPFN
from sklearn.preprocessing import LabelEncoder

# Download if not exists
path_exists = os.path.exists("titanic")

if not path_exists:
    print("No dataset found. Creating 'titanic' directory and downloading.")
    # Create path if it does not exist
    os.makedirs("titanic")
    os.system("kaggle competitions download -c titanic")
    os.system("unzip titanic.zip")
    os.system("mv train.csv test.csv titanic")
    os.system("rm -rf titanic.zip gender_submission.csv")

train = pd.read_csv('titanic/train.csv', header=0, dtype={'Age': np.float64})

for features in ['Age', 'Embarked', 'Cabin', 'Sex', 'Name', 'Ticket', 'Fare']:
    lbl = LabelEncoder()
    lbl.fit(train[features])
    train[features] = lbl.transform(train[features])
train.isnull().sum()


y_target = train['Survived']
train.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1, inplace=True)

adv = Adversarial_TabPFN(X_full=train.to_numpy(), y_full=y_target.to_numpy(), lr=0.05, num_steps=0)
adv.adversarial_attack()
