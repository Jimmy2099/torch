import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

X = torch.randn(100, 1)
y = torch.sin(X)

# minimal example only one parameter weight and bias for testing
model = nn.Linear(in_features=1, out_features=1, bias=True)
print(model)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("weight:", model.weight) #0.6109
print("bias:", model.bias)    # -0.1250

X_test = torch.randn(5, 1)
y_real = torch.sin(X_test).numpy()
y_predictions = model(X_test).detach().numpy()

print("y_result：", y_real)
print("y_prediction:", y_predictions)

plt.figure(figsize=(12, 6))

x_axis = np.arange(y_real.size)
plt.scatter(x_axis, y_real, c='green', label='Real Values')
plt.scatter(x_axis, y_predictions, c='red', label='Predictions')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Real vs Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()
