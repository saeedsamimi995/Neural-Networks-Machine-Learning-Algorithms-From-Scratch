# Deep Learning
# 3 layer AE + 2 layer MLP , Regression

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Pre Processing Functions
def split_train_test_data(X, Y, train_ratio):
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    Y_train = Y[:split_index]
    X_test = X[split_index:]
    Y_test = Y[split_index:]

    return X_train, Y_train, X_test, Y_test

def normalize(list):
    min_val = min(list)
    max_val = max(list)

    return [(x - min_val) / (max_val - min_val) for x in list]

def generate_shifted_series_with_target(data, num_shifts):
    target_shift = 5
    data_array = data.to_numpy().flatten()
    num_samples = len(data_array)

    shifted_data = np.zeros((num_samples, num_shifts + 1))
    shifted_data[:, 0] = data_array
    for shift in range(1, num_shifts + 1):
        shifted_data[:, shift] = np.roll(data_array, shift)

    target = np.roll(data_array, -target_shift)
    target = target[:-target_shift]

    return shifted_data[:-target_shift, :], target

# Additional Functions
def get_activation_function(name):
    if name == "pureline":
        return pureline, pureline_derivative
    elif name == "relu":
        return relu, relu_derivative
    elif name == "leaky relu":
        return leaky_relu, leaky_relu_derivative
    elif name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "tansig":
        return tansig, tansig_derivative
    else:
        raise ValueError("Invalid activation function name")

def initialize_Layer(previous_layer, next_layer, activation_function):
    W = np.random.uniform(-1, 1, size=(next_layer, previous_layer))
    Wb = np.ones((next_layer, 1))
    NET = np.zeros((next_layer, 1))
    F, dF = get_activation_function(activation_function)
    O = np.zeros((next_layer, 1))

    return W, Wb, NET, F, dF, O

# Activation Functions
def pureline(x):
    return x

def pureline_derivative(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    alpha = 0.01
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x):
    alpha = 0.01
    return np.where(x < 0, alpha, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tansig(x):
    return np.tanh(x)

def tansig_derivative(x):
    return 1 - np.square(tansig(x))

# Plot Functions
def plot_progress(epoch, x_train, output_train, mse_train, x_test, output_test, mse_test):
    mse_train = np.array(mse_train)
    mse_test = np.array(mse_test)

    plt.clf()

    plt.subplot(2, 2, 1)
    plt.plot(x_train, color='blue', label='Actual', linewidth=1)
    plt.plot(output_train, color='red', label='Prediction', linewidth=1)
    plt.title('Train')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(mse_train[:epoch+1], color='blue')
    plt.title('MSE - Train')
    plt.xlabel('Epoch')

    plt.subplot(2, 2, 3)
    plt.plot(x_test, color='blue', label='Actual', linewidth=1)
    plt.plot(output_test, color='red', label='Prediction', linewidth=1)
    plt.title('Test')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(mse_test[:epoch+1], color='blue')
    plt.title('MSE - Test')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.pause(0.1)

def plot_regression(x_train, d_train, output_train, x_test, d_test, output_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(x_train[:, 0], d_train, 'b.', label='Actual')
    axes[0].plot(x_train[:, 0], output_train, 'r.', label='Prediction')
    axes[0].set_title('Regression - Train')
    axes[0].set_xlabel('Input')
    axes[0].set_ylabel('Output')
    axes[0].legend()

    m_train, b_train = np.polyfit(x_train[:, 0], output_train.flatten(), 1)
    axes[0].plot(x_train[:, 0], m_train * x_train[:, 0] + b_train, color='black', label='Best Fit', linewidth=3)
    
    axes[1].plot(x_test[:, 0], d_test, 'b.', label='Actual')
    axes[1].plot(x_test[:, 0], output_test, 'r.', label='Prediction')
    axes[1].set_title('Regression - Test')
    axes[1].set_xlabel('Input')
    axes[1].set_ylabel('Output')
    axes[1].legend()

    m_test, b_test = np.polyfit(x_test[:, 0], output_test.flatten(), 1)
    axes[1].plot(x_test[:, 0], m_test * x_test[:, 0] + b_test, color='black', label='Best Fit', linewidth=3)

    plt.tight_layout()
    plt.show()

#############################################     Hyperparameters     #############################################
train_test_ratio = 0.8

# Number of Neurons
input_size = 20
N_en2 = 12
N_en3 = 6
N_mlp1 = 3

# Learning Rate
eta_ae1 = 0.0003
eta_ae2 = 0.0005
eta_ae3 = 0.0007
eta_mlp = 0.0009

# Number of Epochs
epoch_ae1 = 15
epoch_ae2 = 15
epoch_ae3 = 15
epoch_mlp = 21

# Activation Functions
F_encoder_1 = "leaky relu"
F_decoder_1 = "leaky relu"

F_encoder_2 = "leaky relu"
F_decoder_2 = "leaky relu"

F_encoder_3 = "leaky relu"
F_decoder_3 = "leaky relu"

F_mlp_layer_1 = "leaky relu"
F_mlp_layer_2 = "leaky relu"

# Training Options
global_training = False
transpose_training = True
train_bias = False
###################################################################################################################

# Read Data
df = pd.read_csv("daily-minimum-temperatures-in-melbourne.csv")
X = df['Daily minimum temperatures in Melbourne, Australia, 1981-1990'].values
Y = df['Date']

# Split Data
X_train_raw, Y_train, X_test_raw, Y_test = split_train_test_data(X, Y, train_test_ratio)

# Normalization
norm_X_train = normalize(X_train_raw)
norm_X_test = normalize(X_test_raw)

# Shift input
shifted_X_train, target_train = generate_shifted_series_with_target(pd.DataFrame(norm_X_train), input_size - 1)
shifted_X_test, target_test = generate_shifted_series_with_target(pd.DataFrame(norm_X_test), input_size - 1)

# Neural Network Parameters Initialization
X_train = shifted_X_train
X_test = shifted_X_test
d_train = target_train.reshape(-1,1)
d_test = target_test.reshape(-1,1)
num_train = len(X_train)
num_test = len(X_test)

# Number of Neurons in each layer
N_en1 = input_size
N_mlp0 = N_en3
N_mlp2 = 1

# Auto Encoder 1 Initialization
W_en1, Wb_en1, NET_en1, F_en1, dF_en1, O_en1 = initialize_Layer(N_en1, N_en2, F_encoder_1)
W_de1, Wb_de1, NET_de1, F_de1, dF_de1, O_de1 = initialize_Layer(N_en2, N_en1, F_decoder_1)

# Auto Encoder 2 Initialization
W_en2, Wb_en2, NET_en2, F_en2, dF_en2, O_en2 = initialize_Layer(N_en2, N_en3, F_encoder_2)
W_de2, Wb_de2, NET_de2, F_de2, dF_de2, O_de2 = initialize_Layer(N_en3, N_en2, F_decoder_2)

# Auto Encoder 3 Initialization
W_en3, Wb_en3, NET_en3, F_en3, dF_en3, O_en3 = initialize_Layer(N_en3, N_mlp0, F_encoder_3)
W_de3, Wb_de3, NET_de3, F_de3, dF_de3, O_de3 = initialize_Layer(N_mlp0, N_en3, F_decoder_3)

# Multilayer Perceptron Initialization
O_mlp0 = np.zeros((N_mlp0, 1))

W_mlp1, Wb_mlp1, NET_mlp1, F_mlp1, dF_mlp1, O_mlp1 = initialize_Layer(N_mlp0, N_mlp1, F_mlp_layer_1)
W_mlp2, Wb_mlp2, NET_mlp2, F_mlp2, dF_mlp2, O_mlp2 = initialize_Layer(N_mlp1, N_mlp2, F_mlp_layer_2)

# Bias Training Options
if not train_bias:
    Wb_en1 = np.zeros_like(Wb_en1)
    Wb_de1 = np.zeros_like(Wb_de1)
    Wb_en2 = np.zeros_like(Wb_en2)
    Wb_de2 = np.zeros_like(Wb_de2)
    Wb_en3 = np.zeros_like(Wb_en3)
    Wb_de3 = np.zeros_like(Wb_de3)
    Wb_mlp1 = np.zeros_like(Wb_mlp1)
    Wb_mlp2 = np.zeros_like(Wb_mlp2)

MSE_train = np.zeros(epoch_mlp)
MSE_test = np.zeros(epoch_mlp)

# Auto Encoder 1 - Local train
for epoch in range(epoch_ae1):

    for i in range(num_train):

        # Feed Forward
        input = X_train[i, :].reshape(-1,1)

        NET_en1 = np.dot(W_en1,input) + Wb_en1
        O_en1 = F_en1(NET_en1)

        NET_de1 = np.dot(W_de1,O_en1) + Wb_de1
        O_de1 = F_de1(NET_de1)

        error = input - O_de1

        # Back Propagation
        W_de1 += eta_ae1 * np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de1(NET_de1) )).T , O_en1.reshape(1,-1) )

        if transpose_training:
            W_en1 = W_de1.T
        else:
            W_en1 += eta_ae1 * np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de1(NET_de1)) ) , W_de1 ) , np.diagflat(dF_en1(NET_en1)) ).T , input.reshape(1,-1) )
        
        if train_bias:
            Wb_de1 += eta_ae1 * np.dot( error.reshape(1,-1) , np.diagflat(dF_de1(NET_de1)) ).T
            Wb_en1 += eta_ae1 * np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de1(NET_de1)) ) , W_de1 ) , np.diagflat(dF_en1(NET_en1)) ).T

print('Auto Encoder 1 local training completed!')

# Auto Encoder 2 - Local train
for epoch in range(epoch_ae2):

    for i in range(num_train):

        # Feed Forward
        input = X_train[i, :].reshape(-1,1)

        NET_en1 = np.dot(W_en1,input) + Wb_en1
        O_en1 = F_en1(NET_en1)

        NET_en2 = np.dot(W_en2,O_en1) + Wb_en2
        O_en2 = F_en2(NET_en2)

        NET_de2 = np.dot(W_de2,O_en2) + Wb_de2
        O_de2 = F_de2(NET_de2)

        error = O_en1 - O_de2

        # Back Propagation
        W_de2 += eta_ae2 * np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de2(NET_de2)) ).T , O_en2.reshape(1,-1) )

        if transpose_training:
            W_en2 = W_de2.T
        else:
            W_en2 += eta_ae2 * np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de2(NET_de2)) ) , W_de2 ) , np.diagflat(dF_en2(NET_en2)) ).T , O_en1.reshape(1,-1) )

        if train_bias:
            Wb_de2 += eta_ae2 * np.dot( error.reshape(1,-1) , np.diagflat(dF_de2(NET_de2)) ).T
            Wb_en2 += eta_ae2 * np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de2(NET_de2)) ) , W_de2 ) , np.diagflat(dF_en2(NET_en2)) ).T

print('Auto Encoder 2 local training completed!')

# Auto Encoder 3 - Local train
for epoch in range(epoch_ae3):

    for i in range(num_train):

        # Feed Forward
        input = X_train[i, :].reshape(-1,1)

        NET_en1 = np.dot(W_en1,input) + Wb_en1
        O_en1 = F_en1(NET_en1)

        NET_en2 = np.dot(W_en2,O_en1) + Wb_en2
        O_en2 = F_en2(NET_en2)

        NET_en3 = np.dot(W_en3,O_en2) + Wb_en3
        O_en3 = F_en3(NET_en3)

        NET_de3 = np.dot(W_de3,O_en3) + Wb_de3
        O_de3 = F_de3(NET_de3)

        error = O_en2 - O_de3

        # Back Propagation
        W_de3 += eta_ae3 * np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de3(NET_de3)) ).T , O_en3.reshape(1,-1) )

        if transpose_training:
            W_en3 = W_de3.T
        else:
            W_en3 += eta_ae3 * np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de3(NET_de3)) ) , W_de3 ) , np.diagflat(dF_en3(NET_en3)) ).T , O_en2.reshape(1,-1) )

        if train_bias:
            Wb_de3 += eta_ae3 * np.dot( error.reshape(1,-1) , np.diagflat(dF_de3(NET_de3)) ).T
            Wb_en3 += eta_ae3 * np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_de3(NET_de3)) ) , W_de3 ) , np.diagflat(dF_en3(NET_en3)) ).T

print('Auto Encoder 3 local training completed!')

# 2 Layer Perceptron
for epoch in range(epoch_mlp):

    # Train
    output_train = np.zeros(num_train)
    error_train = np.zeros(num_train)
    
    for i in range(num_train):

        # Feed Forward
        input = X_train[i, :].reshape(-1,1)

        NET_en1 = np.dot(W_en1,input) + Wb_en1
        O_en1 = F_en1(NET_en1)

        NET_en2 = np.dot(W_en2,O_en1) + Wb_en2
        O_en2 = F_en2(NET_en2)

        NET_en3 = np.dot(W_en3,O_en2) + Wb_en3
        O_en3 = F_en3(NET_en3)

        O_mlp0 = O_en3

        NET_mlp1 = np.dot(W_mlp1,O_en3) + Wb_mlp1
        O_mlp1 = F_mlp1(NET_mlp1)

        NET_mlp2 = np.dot(W_mlp2,O_mlp1) + Wb_mlp2
        O_mlp2 = F_mlp2(NET_mlp2)

        output_train[i] = O_mlp2.item()
        error = d_train[i] - output_train[i]
        error_train[i] = error.item()

        # Back Propagation
        W_mlp2 += eta_mlp * np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ).T , O_mlp1.reshape(1,-1) )
        W_mlp1 += eta_mlp * np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ).T , O_mlp0.reshape(1,-1) )

        if global_training:
            W_en3 += eta_ae3 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ).T , O_en2.reshape(1,-1) )
            W_en2 += eta_ae2 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ).T , O_en1.reshape(1,-1) )
            W_en1 += eta_ae1 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) ,W_mlp1  ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ) , W_en2 ) , np.diagflat(dF_en1(NET_en1)) ).T , input.reshape(1,-1) )

        if train_bias:
            Wb_mlp2 += eta_mlp * np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ).T
            Wb_mlp1 += eta_mlp * np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ).T

            if global_training:
                Wb_en3 += eta_ae3 * np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ).T
                Wb_en2 += eta_ae2 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ).T
                Wb_en1 += eta_ae1 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( error.reshape(1,-1) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) ,W_mlp1  ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ) , W_en2 ) , np.diagflat(dF_en1(NET_en1)) ).T

    MSE_train[epoch] = np.mean(np.square(error_train))

    # Test
    output_test = np.zeros(num_test)
    error_test = np.zeros(num_test)

    for i in range(num_test):

        # Feed Forward
        input = X_test[i, :].reshape(-1,1)

        NET_en1 = np.dot(W_en1,input) + Wb_en1
        O_en1 = F_en1(NET_en1)

        NET_en2 = np.dot(W_en2,O_en1) + Wb_en2
        O_en2 = F_en2(NET_en2)

        NET_en3 = np.dot(W_en3,O_en2) + Wb_en3
        O_en3 = F_en3(NET_en3)

        O_mlp0 = O_en3

        NET_mlp1 = np.dot(W_mlp1,O_en3) + Wb_mlp1
        O_mlp1 = F_mlp1(NET_mlp1)

        NET_mlp2 = np.dot(W_mlp2,O_mlp1) + Wb_mlp2
        O_mlp2 = F_mlp2(NET_mlp2)

        output_test[i] = O_mlp2.item()
        error = d_test[i] - output_test[i]
        error_test[i] = error.item()
    
    MSE_test[epoch] = np.mean(np.square(error_test))

    plot_progress(epoch, X_train[:,0], output_train, MSE_train.reshape(-1,1), X_test[:,0], output_test, MSE_test.reshape(-1,1))

plt.show()
print('2 Layer Perceptron training completed!')

plot_regression(X_train, d_train, output_train, X_test, d_test, output_test)

print("Final Train MSE: = ", MSE_train[-1])
print("Final Test MSE = ", MSE_test[-1])