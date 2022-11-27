import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# check if path exists
path_exists = os.path.exists("titanic")

if not path_exists:
    print("No dataset found. Creating 'titanic' directory and downloading.")
    # Create path if it does not exist
    os.makedirs("titanic")
    # Download dataset if it is not already there
    os.system("kaggle competitions download -c titanic")
    os.system("unzip titanic.zip")
    os.system("mv train.csv test.csv titanic")
    os.system("rm -rf titanic.zip gender_submission.csv")

# Load training dataframe
train = pd.read_csv('titanic/train.csv', header=0)

for features in ['Age', 'Embarked', 'Cabin', 'Sex', 'Fare']:
    lbl = LabelEncoder()
    lbl.fit(train[features])
    train[features] = lbl.transform(train[features])
train.isnull().sum()

y_target = train['Survived']
train.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1, inplace=True)

if not os.path.exists('titanic/'):
    os.mkdir('titanic/')

np.save(f"titanic/X", train.to_numpy())
np.save(f"titanic/y", y_target.to_numpy())
