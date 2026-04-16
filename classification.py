import math


# Training data
# Feature = hours of suspicious activity
# Label = 1 (spam) or 0 (not spam)
activity = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
labels =   [0,   0,   0,   0,   1,   1,   1,   1  ]

def sigmoid(x):
    return 1/(1+math.exp(-x))

#random w and b
w = 0
b = 0

# learning rate
alpha = 0.1

#Epochs
epochs = 10000

#Training data or Examples
n = len(activity)


#Training Loop
for epoch in range(epochs):
    total_loss = 0
    grad_w = 0
    grad_b = 0

    for i in range(n):
        #Prediction wrapped in sigmoid
        linear = w * activity[i]+b
        prediction = sigmoid(linear)

        error = prediction-labels[i]

        #gradients
        grad_w +=error*activity[i]
        grad_b+=error

        total_loss += -(labels[i] * math.log(prediction) + (1 - labels[i]) * math.log(1 - prediction))
    
    grad_w = grad_w/n
    grad_b = grad_b/n
    total_loss = total_loss/n

    w = w-alpha*grad_w
    b = b-alpha*grad_b

    if epoch % 500 == 0:
        print(f"Epoch {epoch} → Loss: {total_loss:.4f} → w: {w:.2f} → b: {b:.2f}")

#Test predtictions
test = [1.0,2.0,3.0,4.0]

for t in test:
    linear = w*t+b
    prob = sigmoid(linear)
    label = 1 if prob>0.5 else 0
    print(f"Activity: {t} → Probability: {prob:.2f} → {'Spam' if label == 1 else 'Not Spam'}")


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

plt.scatter(activity, labels, label="Actual Data")
predicted =[sigmoid(w*a + b) for a in activity]
plt.plot(activity, predicted, label="Model Prediction")

plt.xlabel("Suspicious Activity")
plt.ylabel("Probability Of Spam")
plt.title("Logistic Regression (Sigmoid Curve)")
plt.legend()
plt.grid(True)

plt.savefig("logistic_regression.png")
plt.show()