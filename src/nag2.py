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

learning_rate = 0.001
epochs = 100000
v_a = 0
v_b = 0
v_c = 0
beta = 0.9

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    a_star = a - learning_rate * v_a
    b_star = b - learning_rate * v_b
    c_star = c - learning_rate * v_c

    y_pred = a_star * x_i**2 + b_star * x_i + c_star
    error = (y_i - y_pred)

    da =  error * -1 * x_i**2
    db =  error * -1 * x_i
    dc =  error * -1 * 1

    v_a = beta * v_a + learning_rate * da
    v_b = beta * v_b + learning_rate * db
    v_c = beta * v_c + learning_rate * dc

    a -= learning_rate * v_a
    b -= learning_rate * v_b
    c -= learning_rate * v_c

mse = np.mean((y - (a * x**2 + b * x + c))**2)
print("Trained weight:", a,b,c)
print("NAG MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = a * x**2 + b * x + c
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nNAG')
plt.ylabel('Y')

plt.legend()
plt.show()

