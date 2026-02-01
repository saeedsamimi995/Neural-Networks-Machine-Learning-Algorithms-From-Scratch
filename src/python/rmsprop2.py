import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data from google drive
data = pd.read_csv('/content/drive/MyDrive/NN_HW1(data2).csv')
x = data.iloc[:,0]
y = data.iloc[:,1]

a = np.random.rand(1)
b = np.random.rand(1)
c = np.random.rand(1)

learning_rate = 0.01
epochs = 100000
epsilon = 1e-7
beta = 0.9
S_a = 0
S_b = 0
S_c = 0

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = a * x_i**2 + b * x_i + c
    error = (y_i - y_pred)

    da =  error * -1 * x_i**2
    db =  error * -1 * x_i
    dc =  error * -1 * 1

    S_a += beta * S_a + (1 - beta) * (da**2)
    S_b += beta * S_b + (1 - beta) * (db**2)
    S_c += beta * S_c + (1 - beta) * (dc**2)

    a -= (learning_rate / (np.sqrt(S_a) + epsilon)) * da
    b -= (learning_rate / (np.sqrt(S_b) + epsilon)) * db
    c -= (learning_rate / (np.sqrt(S_c) + epsilon)) * dc

mse = np.mean((y - (a * x**2 + b * x + c))**2)
print("Trained weight:", a,b,c)
print("RMSprop MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = a * x**2 + b * x + c
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nRMSprop')
plt.ylabel('Y')

plt.legend()
plt.show()

