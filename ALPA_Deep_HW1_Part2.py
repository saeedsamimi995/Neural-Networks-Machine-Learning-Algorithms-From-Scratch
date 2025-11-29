# Deep Learning
# HW 1 - Part 2
# Ali Pazouki - 40203334
# 3 layer AE + 2 layer MLP , Classification

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE # for Data Augmentation

# Pre Processing Functions
def split_train_test_data(X, Y, train_ratio):
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    Y_train = Y[:split_index]
    X_test = X[split_index:]
    Y_test = Y[split_index:]

    return X_train, Y_train, X_test, Y_test

def normalize(matrix):
    normalized_matrix = []
    for col in matrix.T:
        min_val = min(col)
        max_val = max(col)
        normalized_col = [(x - min_val) / (max_val - min_val) if max_val != min_val else x for x in col]
        normalized_matrix.append(normalized_col)
    return np.array(normalized_matrix).T

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

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def softmax_derivative(x):
    softmax_values = softmax(x).reshape(-1,1)
    return np.diag(softmax_values.flatten()) - np.dot(softmax_values, softmax_values.T)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred + epsilon))

def cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)    
    return -y_true / y_pred

# Plot Functions
def plot_mse(epoch, mse_train, mse_test):
    mse_train = np.array(mse_train)
    mse_test = np.array(mse_test)

    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(mse_train[:epoch+1], color='blue')
    plt.title('MSE - Train')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(mse_test[:epoch+1], color='blue')
    plt.title('MSE - Test')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.pause(0.1)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    # Calculate Confusion Matrix
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    num_classes = y_true.shape[1]
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true_labels)):
        cm[y_true_labels[i], y_pred_labels[i]] += 1

    # Plot
    cmap = plt.cm.Blues
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            num_text = format(cm[i, j], 'd')
            perc_text = format((cm[i, j] / np.sum(cm[i])) * 100, '.2f') + "%"
            plt.text(j, i, f"{num_text}\n({perc_text})",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#############################################     Hyperparameters     #############################################
train_test_ratio = 0.8

# Number of Neurons
N_en2 = 35
N_en3 = 20
N_mlp1 = 5

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
global_training = True
transpose_training = False
train_bias = False
Data_Augmentation = True
###################################################################################################################

# Read Data
data = pd.read_csv('eighthr.data.csv', usecols=range(1, 74), na_values='?', header=None)
data.dropna(subset=range(1, 74), inplace=True)
num_classes = data.iloc[:, -1].nunique()

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

class_counts = data.iloc[:, -1].value_counts()
class_0 = class_counts[0]
class_1 = class_counts[1]

# Shuffle Data
data_shuffled = data.sample(frac=1)
X = data_shuffled.iloc[:, :-1].values
Y = data_shuffled.iloc[:, -1].values.reshape(-1,1)

# Data Augmentation
if Data_Augmentation:
    # Generate synthetic samples
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)

    unique, counts = np.unique(Y_resampled, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("Class distribution after augmentation:", class_distribution)

    # Shuffle resampled data
    combined_data = np.column_stack((X_resampled, Y_resampled))
    np.random.shuffle(combined_data)
    X_resampled_shuffled = combined_data[:, :-1]
    Y_resampled_shuffled = combined_data[:, -1].reshape(-1,1)

    X, Y = X_resampled_shuffled, Y_resampled_shuffled

# Split Data
X_train_raw, Y_train, X_test_raw, Y_test = split_train_test_data(X, Y, train_test_ratio)

# Normalization
norm_X_train = normalize(X_train_raw)
norm_X_test = normalize(X_test_raw)

# One-hot encoding
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

Y_train_one_hot = np.eye(num_classes)[Y_train.flatten()]
Y_test_one_hot = np.eye(num_classes)[Y_test.flatten()]

# Neural Network Parameters Initialization
X_train = norm_X_train
X_test = norm_X_test
d_train = Y_train_one_hot
d_test = Y_test_one_hot
num_train = len(X_train)
num_test = len(X_test)

# Number of Neurons in each layer
input_size = X.shape[1]
N_en1 = input_size
N_mlp0 = N_en3
N_mlp2 = num_classes

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

Z = np.zeros((N_mlp2, 1))

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
    output_train = np.zeros((num_train,num_classes))
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

        Z = softmax(O_mlp2)

        output_train[i] = Z.reshape(-1)
        error_train[i] = cross_entropy(d_train[i], output_train[i])

        # Back Propagation
        e = - cross_entropy_derivative(d_train[i], output_train[i]).reshape(1,-1)
        W_mlp2 += eta_mlp * np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ).T , O_mlp1.reshape(1,-1) )
        W_mlp1 += eta_mlp * np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2)) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ).T , O_mlp0.reshape(1,-1) )
        
        if global_training:
            W_en3 += eta_ae3 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ).T , O_en2.reshape(1,-1) )
            W_en2 += eta_ae2 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ).T , O_en1.reshape(1,-1) )
            W_en1 += eta_ae1 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) ,W_mlp1  ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ) , W_en2 ) , np.diagflat(dF_en1(NET_en1)) ).T , input.reshape(1,-1) )

        if train_bias:
            Wb_mlp2 += eta_mlp * np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ).T
            Wb_mlp1 += eta_mlp * np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ).T

            if global_training:
                Wb_en3 += eta_ae3 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ).T
                Wb_en2 += eta_ae2 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) , W_mlp1 ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ).T
                Wb_en1 += eta_ae1 * np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( e , softmax_derivative(O_mlp2) ) , np.diagflat(dF_mlp2(NET_mlp2)) ) , W_mlp2 ) , np.diagflat(dF_mlp1(NET_mlp1)) ) ,W_mlp1  ) , np.diagflat(dF_en3(NET_en3)) ) , W_en3 ) , np.diagflat(dF_en2(NET_en2)) ) , W_en2 ) , np.diagflat(dF_en1(NET_en1)) ).T

    MSE_train[epoch] = np.mean(np.square(error_train))
    
    # Test
    output_test = np.zeros((num_test,num_classes))
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

        Z = softmax(O_mlp2)

        output_test[i] = Z.reshape(-1)
        error_test[i] = cross_entropy(d_train[i], output_train[i])
    
    MSE_test[epoch] = np.mean(np.square(error_test))

    plot_mse(epoch, MSE_train.reshape(-1,1), MSE_test.reshape(-1,1))

plt.show()
print('2 Layer Perceptron training completed!')

plot_confusion_matrix(d_train, output_train, ['Class 0', 'Class 1'], title='Train Confusion Matrix')
plot_confusion_matrix(d_test, output_test, ['Class 0', 'Class 1'], title='Test Confusion Matrix')

print("Final Train MSE: = ", MSE_train[-1])
print("Final Test MSE = ", MSE_test[-1])