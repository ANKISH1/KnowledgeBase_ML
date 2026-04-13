
# Our training data
# Feature = hours studied, Label = marks scored
hours =  [1, 2, 3, 4, 5, 6, 7, 8]
marks =  [10, 22, 28, 42, 48, 61, 70, 79]

# Start with random w and b
w = 0.0
b = 0.0

# Learning rate
alpha = 0.01

# Number of training steps
epochs = 1000

# Number of training examples
n = len(hours)

# Training loop
for epoch in range(epochs):
    
    # Step 1: Make predictions and calculate loss
    total_loss = 0
    
    for i in range(n):
        prediction = w * hours[i] + b
        error = prediction - marks[i]
        total_loss += error ** 2
    
    mse = total_loss / n
    
    grad_w = 0
    grad_b = 0

    for i in range(n):
        prediction = w*hours[i]+b
        error = prediction-marks[i]
        grad_w+=error*hours[i]
        grad_b +=error

    grad_w = grad_w/n
    grad_b = grad_b/n

    w = w-alpha*grad_w
    b = b-alpha*grad_b

    if epoch%100 == 0:
        print(f"Epoch {epoch} -> Loss: {mse:.2f} -> w: {w:.2f} -> b:{b:.2f}")    

# Make a prediction on new data
hours_studied = 9
predicted_marks = w * hours_studied + b
print(f"\nIf you study {hours_studied} hours → predicted marks: {predicted_marks:.1f}") 


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))  # FIRST

plt.scatter(hours, marks, label="Actual Data")
predicted = [w*h + b for h in hours]
plt.plot(hours, predicted, color = 'red', label="Model Prediction")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)

plt.savefig("linear_regression.png")  # clean image for LinkedIn
plt.show()