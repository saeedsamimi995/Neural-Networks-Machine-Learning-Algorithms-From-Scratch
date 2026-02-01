import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data from google drive
data = pd.read_csv('/content/drive/MyDrive/NN_HW1(data).csv')
x = data.iloc[:,0]
y = data.iloc[:,1]

w = np.random.rand(1)
b = np.random.rand(1)

epochs = 100000
learning_rate = 0.01
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999
v_w = 0
v_b = 0
S_w = 1e-8
S_b = 1e-8
t = 0

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = w * x_i + b
    error = (y_i - y_pred)

    dw =  error * -1 * x_i
    db =  error * -1 * 1

    t += 1

    v_w = beta1 * v_w + (1 - beta1) * dw
    v_b = beta1 * v_b + (1 - beta1) * db

    S_w = beta2 * S_w + (1 - beta2) * (dw**2)
    S_b = beta2 * S_b + (1 - beta2) * (db**2)

    v_w_hat = v_w / (1 - beta1**t)
    v_b_hat = v_b / (1 - beta1**t)

    S_w_hat = S_w / (1 - beta2**t)
    S_b_hat = S_b / (1 - beta2**t)

    w -= (learning_rate / (np.sqrt(S_w_hat) + epsilon)) * v_w_hat
    b -= (learning_rate / (np.sqrt(S_b_hat) + epsilon)) * v_b_hat

mse = np.mean((y - (w * x + b))**2)
print("Trained weight:", w)
print("Trained bias:", b)
print("adam MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = w * x + b
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nadam')
plt.ylabel('Y')

plt.legend()
plt.show()