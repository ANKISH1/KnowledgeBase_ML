from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# My data
# Features: [Weather, Temperature]
# 0 = Sunny, 1 = Rainy
# 0 = Mild, 1 = Hot

X = [
    [0, 1],  # Sunny, Hot
    [0, 0],  # Sunny, Mild
    [1, 0],  # Rainy, Mild
    [0, 0],  # Sunny, Mild
    [1, 1],  # Rainy, Hot
    [0, 0],  # Sunny, Mild
    [1, 0],  # Rainy, Mild
    [0, 1],  # Sunny, Hot
]

# Labels: 1 = Play, 0 = Don't Play
y = [0, 1, 0, 1, 0, 1, 0, 0]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

#Make Predictions
predictions = model.predict(X_test)

#Evaluate
print(f"Predictions: {predictions}")
print(f"Actual: {y_test}")
print(f"Accuracy: {accuracy_score(y_test, predictions): .2f}")
print("\n", classification_report(y_test, predictions))
