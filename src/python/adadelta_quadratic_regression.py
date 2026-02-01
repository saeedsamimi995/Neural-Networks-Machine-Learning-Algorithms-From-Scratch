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
epsilon = 1e-6
beta = 0.9
S_da = 0
S_db = 0
S_dc = 0
D_a = 0
D_b = 0
D_c = 0
a_t = 0
b_t = 0
c_t = 0
D_a_t = 0
D_b_t = 0
D_c_t = 0

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = a * x_i**2 + b * x_i + c
    error = (y_i - y_pred)

    da =  error * -1 * x_i**2
    db =  error * -1 * x_i
    dc =  error * -1 * 1

    S_da = beta * S_da + (1 - beta) * (da**2)
    S_db = beta * S_db + (1 - beta) * (db**2)
    S_dc = beta * S_dc + (1 - beta) * (dc**2)

    delta_a = a - a_t
    delta_b = b - b_t
    delta_c = c - c_t

    D_a = beta * D_a_t + (1-beta) * (delta_a**2)
    D_b = beta * D_b_t + (1-beta) * (delta_b**2)
    D_c = beta * D_c_t + (1-beta) * (delta_c**2)

    D_a_t = D_a
    D_b_t = D_b
    D_c_t = D_c

    a -= np.sqrt(D_a + epsilon) / np.sqrt(S_da + epsilon) * da
    b -= np.sqrt(D_b + epsilon) / np.sqrt(S_db + epsilon) * db
    c -= np.sqrt(D_c + epsilon) / np.sqrt(S_dc + epsilon) * dc

    a_t = a
    b_t = b
    c_t = c

mse = np.mean((y - (a * x**2 + b * x + c))**2)
print("Trained weight:", a,b,c)
print("adadelta MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = a * x**2 + b * x + c
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nadadelta')
plt.ylabel('Y')

plt.legend()
plt.show()

