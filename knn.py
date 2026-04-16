from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Features: [hours studied, sleep hours]
X = [
    [1, 4], [2, 5], [3, 6], [4, 7],
    [5, 8], [6, 7], [7, 8], [8, 9],
    [2, 3], [3, 4], [6, 6], [7, 7]
]

# Labels: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

#Predict
predictions = model.predict(X_test)

# Evaluate
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test}")
print(f"Accuracy:    {accuracy_score(y_test, predictions):.2f}")