
import torch
import torch.nn as nn

# Features: [hours studied, sleep hours]
X = torch.tensor([
    [1.0, 4.0],
    [2.0, 5.0],
    [3.0, 6.0],
    [4.0, 7.0],
    [5.0, 8.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0],
], dtype=torch.float32)

# Labels: 0 = Fail, 1 = Pass
y = torch.tensor([
    [0], [0], [0], [0],
    [1], [1], [1], [1]
], dtype=torch.float32)

# Define the Neural Network
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
    
# Create Model
model = StudentNet()

# Loss function and optimizer
criterion = nn.BCELoss() #Binary Cross Entropy Loss..similar to Logistic Regression
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # SGD = Stochastic Gradient Descent. lr = learning rate..again similar

# training loop
for epoch in range(1000):
    predictions = model(X) #Make predictions
    loss = criterion(predictions,y) #Calculate Loss

    optimizer.zero_grad() #reset gradients to 0 (we did this manually)
    loss.backward() #calculate all gradients automatically (autograd)
    optimizer.step() #update all weights (we did w = w - alpha * grad_w)
 
    if epoch % 100 ==0:
        print(f"Epoch {epoch} -> Loss: {loss.item():.4f}")


#Test
with torch.no_grad():
    test = torch.tensor([[9.0, 9.0], [1.0, 3.0]], dtype=torch.float32)
    output = model(test)
    for i, o in enumerate(output):
        print(f"Student {i+1} → Probability: {o.item():.2f} → {'Pass' if o.item() > 0.5 else 'Fail'}")