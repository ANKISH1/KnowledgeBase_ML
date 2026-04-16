from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Data

X = [
    [1, 4], [2, 5], [3, 6], [4, 7],
    [5, 8], [6, 7], [7, 8], [8, 9],
    [2, 3], [3, 4], [6, 6], [7, 7]
]

y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Predictions: {predictions}")
print(f"Actual: {y_test}")

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"Support Vectors: {model.support_vectors_}")