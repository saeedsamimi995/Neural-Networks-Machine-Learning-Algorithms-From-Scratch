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
epsilon = 1e-6
beta = 0.95
S_dw = 0
S_db = 0
D_w = 0
D_b = 0

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = w * x_i + b
    error = (y_i - y_pred)

    
    dw =  error * -1 * x_i
    db =  error * -1 * 1

    S_dw = beta * S_dw + (1 - beta) * (dw**2)
    S_db = beta * S_db + (1 - beta) * (db**2)

    delta_w = w - w_t
    delta_b = b - b_t

    D_w = beta * D_w_t + (1-beta) * (delta_w**2)
    D_b = beta * D_b_t + (1-beta) * (delta_b**2)

    D_w_t = D_w
    D_b_t = D_b 

    w -= np.sqrt(D_w + epsilon) / np.sqrt(S_dw + epsilon) * dw
    b -= np.sqrt(D_b + epsilon) / np.sqrt(S_db + epsilon) * db
    
    w_t = w
    b_t = b
    

mse = np.mean((y - (w * x + b))**2)
print("Trained weight:", w)
print("Trained bias:", b)
print("adadelta MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = w * x + b
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nadadelta')
plt.ylabel('Y')

plt.legend()
plt.show()