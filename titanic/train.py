import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn. model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Load and preprocess data
df = pd.read_csv('titanic.csv')

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#Saperate features and label
X = df.drop('Survived', axis=1)
y = df['Survived']


#Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

#Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))

# Feature importance
import pandas as pd
features = X.columns
importances = model.feature_importances_

for feature, importance in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.3f}")