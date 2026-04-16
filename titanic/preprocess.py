import pandas as pd

df = pd.read_csv('titanic.csv')

#Step1 Drop useless columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#Step2 Fill missing age with average age
df['Age'] = df['Age'].fillna(df['Age'].mean())

#Step3 Fill Missing embarked with most common value

df['Embarked']  = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'male': 0, 'female':1})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

#Check Missing Values again
print(df.head())
print(df.isnull().sum())
print(df.shape)