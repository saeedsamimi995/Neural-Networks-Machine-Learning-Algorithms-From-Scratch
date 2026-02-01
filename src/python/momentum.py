import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data from google drive
data = pd.read_csv('/content/drive/MyDrive/NN_HW1(data).csv')
x = data.iloc[:,0]
y = data.iloc[:,1]

w = np.random.rand(1)
b = np.random.rand(1)

learning_rate = 0.01
epochs = 100000
v_w = 0
v_b = 0
beta = 0.9

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = w * x_i + b
    error = (y_i - y_pred)

    dw =  error * -1 * x_i
    db =  error * -1 * 1

    w = w - learning_rate * v_w
    b = b - learning_rate * v_b
    
    v_w = beta * v_w + (1-beta) * dw
    v_b = beta * v_b + (1-beta) * db

mse = np.mean((y - (w * x + b))**2)
print("Trained weight:", w)
print("Trained bias:", b)
print("Momentum MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = w * x + b
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nMomentum')
plt.ylabel('Y')

plt.legend()
plt.show()
