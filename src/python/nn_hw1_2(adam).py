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
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999
v_a = 0
v_b = 0
v_c = 0
S_a = 1e-8
S_b = 1e-8
S_c = 1e-8
t = 0

for epoch in range(epochs):

    random_index = np.random.randint(len(x))

    x_i = x[random_index]
    y_i = y[random_index]

    y_pred = a * x_i**2 + b * x_i + c
    error = (y_i - y_pred)

    da =  error * -1 * x_i**2
    db =  error * -1 * x_i
    dc =  error * -1 * 1

    t += 1

    v_a = beta1 * v_a + (1 - beta1) * da
    v_b = beta1 * v_b + (1 - beta1) * db
    v_c = beta1 * v_c + (1 - beta1) * dc

    S_a = beta2 * S_a + (1 - beta2) * (da**2)
    S_b = beta2 * S_b + (1 - beta2) * (db**2)
    S_c = beta2 * S_c + (1 - beta2) * (dc**2)

    v_a_hat = v_a / (1 - beta1**t)
    v_b_hat = v_b / (1 - beta1**t)
    v_c_hat = v_c / (1 - beta1**t)

    S_a_hat = S_a / (1 - beta2**t)
    S_b_hat = S_b / (1 - beta2**t)
    S_c_hat = S_c / (1 - beta2**t)

    a -= (learning_rate / (np.sqrt(S_a_hat) + epsilon)) * v_a_hat
    b -= (learning_rate / (np.sqrt(S_b_hat) + epsilon)) * v_b_hat
    c -= (learning_rate / (np.sqrt(S_c_hat) + epsilon)) * v_c_hat

mse = np.mean((y - (a * x**2 + b * x + c))**2)
print("Trained weight:", a,b,c)
print("adam MSE:", mse)

plt.scatter(x, y, label='Data Points', color='blue')

true_line = a * x**2 + b * x + c
plt.plot(x, true_line, label='True Line', color='red', linestyle='-', linewidth=3)

plt.xlabel('X\nadam')
plt.ylabel('Y')

plt.legend()
plt.show()

