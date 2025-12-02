import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def sigmoid(x):
    return 1 / (1 + math.e ** (-1 * x))


def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a, (-1, 1))
    b = 1 - sigmoid(x)
    b = np.reshape(b, (-1, 1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a, b)))


# define a normalization function
def normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized = (column - min_val) / (max_val - min_val)
    return normalized


split_ratio = 0.7
eta = 0.3
epochs = 100

data = pd.read_csv('dataset/Bias_correction_ucl_mean.csv')

# apply normalization to each column
data = data.apply(normalize)
data = np.array(data)

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = data[:split_line_number, :20]
x_test = data[split_line_number:, :20]
y_train = data[:split_line_number, 20]
y_test = data[split_line_number:, 20]

input_dimension = np.shape(x_train)[1]
l1_ae_neurons = 25
l2_ae_neurons = 13
l3_ae_neurons = 7
l1_mlp_neurons = 9
l2_mlp_neurons = 7
l3_mlp_neurons = 1

we1 = np.random.uniform(low=-1, high=1, size=(l1_ae_neurons, input_dimension))
we2 = np.random.uniform(low=-1, high=1, size=(l2_ae_neurons, l1_ae_neurons))
we3 = np.random.uniform(low=-1, high=1, size=(l3_ae_neurons, l2_ae_neurons))

wd1 = np.random.uniform(low=-1, high=1, size=(input_dimension, l1_ae_neurons))
wd2 = np.random.uniform(low=-1, high=1, size=(l1_ae_neurons, l2_ae_neurons))
wd3 = np.random.uniform(low=-1, high=1, size=(l2_ae_neurons, l3_ae_neurons))

w1 = np.random.uniform(low=-1, high=1, size=(l1_mlp_neurons, l3_ae_neurons))
w2 = np.random.uniform(low=-1, high=1, size=(l2_mlp_neurons, l1_mlp_neurons))
w3 = np.random.uniform(low=-1, high=1, size=(l3_mlp_neurons, l2_mlp_neurons))

MSE_train = []
MSE_test = []
train_loss = []
test_loss = []

# AE layer one local train
for i in range(epochs):

    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    h1 = np.zeros((l1_ae_neurons, 1))

    x_hat = np.zeros((input_dimension, 1))

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_train[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 1 Decoder
        net_d1 = np.matmul(wd1, h1)
        x_hat = sigmoid(net_d1)
        x_hat = np.reshape(x_hat, (-1, 1))

        output_train.append(x_hat)

        # Error
        err = y_train[j] - x_hat
        sqr_err_epoch_train.append(err ** 2)

        # Back propagation
        f_d1_deriviate = sigmoid_deriviate(net_d1)
        f_e1_deriviate = sigmoid_deriviate(net_e1)

        err_f_d1_deriviate = np.matmul(np.transpose(err), f_d1_deriviate)
        err_f_d1_deriviate_h1 = np.matmul(np.transpose(err_f_d1_deriviate), np.transpose(h1))

        err_f_d1_deriviate_wd1 = np.matmul(err_f_d1_deriviate, wd1)
        err_f_d1_deriviate_wd1_f_e1_deriviate = np.matmul(err_f_d1_deriviate_wd1, f_e1_deriviate)
        err_f_d1_deriviate_wd1_f_e1_deriviate_x = np.matmul(np.transpose(err_f_d1_deriviate_wd1_f_e1_deriviate),
                                                            np.reshape(x_train[j], (1, -1)))

        we1 = np.add(we1, eta * err_f_d1_deriviate_wd1_f_e1_deriviate_x)
        wd1 = np.add(wd1, eta * err_f_d1_deriviate_h1)

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train)) / np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_test[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 1 Decoder
        net_d1 = np.matmul(wd1, h1)
        x_hat = sigmoid(net_d1)
        x_hat = np.reshape(x_hat, (-1, 1))

        output_test.append(x_hat)

        # Error
        err = y_test[j] - x_hat
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test)) / np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)

train_loss = np.squeeze(np.array(MSE_train)).mean(axis=1)
test_loss = np.squeeze(np.array(MSE_test)).mean(axis=1)

# plot the train and test loss
plt.plot(range(epochs), train_loss, 'b', label='Train Loss')
plt.plot(range(epochs), test_loss, 'r', label='Test Loss')
plt.title('Train and Test Loss of Autoencoder')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("==============================================================================================================")
print("==============================================================================================================")
print("Train mean mse for each epoch on AE one")
print(np.mean(np.squeeze(np.asarray(MSE_train)), axis=1))
print("==============================================================================================================")
print("Test mean mse for each epoch on AE one")
print(np.mean(np.squeeze(np.asarray(MSE_test)), axis=1))

MSE_train = []
MSE_test = []

# AE layer two local train
for i in range(epochs):

    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    h1 = np.zeros((l1_ae_neurons, 1))
    h2 = np.zeros((l2_ae_neurons, 1))

    h1_hat = np.zeros((l1_ae_neurons, 1))

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_train[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 2 Decoder
        net_d2 = np.matmul(wd2, h2)
        h1_hat = sigmoid(net_d2)
        h1_hat = np.reshape(h1_hat, (-1, 1))

        output_train.append(h1_hat)

        # Error
        err = y_train[j] - h1_hat
        sqr_err_epoch_train.append(err ** 2)

        # Back propagation
        f_d2_deriviate = sigmoid_deriviate(net_d2)
        f_e2_deriviate = sigmoid_deriviate(net_e2)

        err_f_d2_deriviate = np.matmul(np.transpose(err), f_d2_deriviate)
        err_f_d2_deriviate_h2 = np.matmul(np.transpose(err_f_d2_deriviate), np.transpose(h2))

        err_f_d2_deriviate_wd2 = np.matmul(err_f_d2_deriviate, wd2)
        err_f_d2_deriviate_wd2_f_e2_deriviate = np.matmul(err_f_d2_deriviate_wd2, f_e2_deriviate)
        err_f_d2_deriviate_wd2_f_e2_deriviate_h1 = np.matmul(np.transpose(err_f_d2_deriviate_wd2_f_e2_deriviate),
                                                             np.transpose(h1))

        we2 = np.add(we2, eta * err_f_d2_deriviate_wd2_f_e2_deriviate_h1)
        wd2 = np.add(wd2, eta * err_f_d2_deriviate_h2)

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train)) / np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_test[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 2 Decoder
        net_d2 = np.matmul(wd2, h2)
        h1_hat = sigmoid(net_d2)
        h1_hat = np.reshape(h1_hat, (-1, 1))

        output_test.append(h1_hat)

        # Error
        err = y_test[j] - h1_hat
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test)) / np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)

train_loss = np.squeeze(np.array(MSE_train)).mean(axis=1)
test_loss = np.squeeze(np.array(MSE_test)).mean(axis=1)

# plot the train and test loss
plt.plot(range(epochs), train_loss, 'b', label='Train Loss')
plt.plot(range(epochs), test_loss, 'r', label='Test Loss')
plt.title('Train and Test Loss of Autoencoder')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("==============================================================================================================")
print("==============================================================================================================")
print("Train mean mse for each epoch on AE two")
print(np.mean(np.squeeze(np.asarray(MSE_train)), axis=1))
print("==============================================================================================================")
print("Test mean mse for each epoch on AE two")
print(np.mean(np.squeeze(np.asarray(MSE_test)), axis=1))

MSE_train = []
MSE_test = []

# AE layer three local train
for i in range(epochs):

    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    h1 = np.zeros((l1_ae_neurons, 1))
    h2 = np.zeros((l2_ae_neurons, 1))
    h3 = np.zeros((l3_ae_neurons, 1))

    h2_hat = np.zeros((l2_ae_neurons, 1))

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_train[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 3 Encoder
        net_e3 = np.matmul(we3, h2)
        h3 = sigmoid(net_e3)
        h3 = np.reshape(h3, (-1, 1))

        # Layer 3 Decoder
        net_d3 = np.matmul(wd3, h3)
        h2_hat = sigmoid(net_d3)
        h2_hat = np.reshape(h2_hat, (-1, 1))

        output_train.append(h2_hat)

        # Error
        err = y_train[j] - h2_hat
        sqr_err_epoch_train.append(err ** 2)

        # Back propagation
        f_d3_deriviate = sigmoid_deriviate(net_d3)
        f_e3_deriviate = sigmoid_deriviate(net_e3)

        err_f_d3_deriviate = np.matmul(np.transpose(err), f_d3_deriviate)
        err_f_d3_deriviate_h3 = np.matmul(np.transpose(err_f_d3_deriviate), np.transpose(h3))

        err_f_d3_deriviate_wd3 = np.matmul(err_f_d3_deriviate, wd3)
        err_f_d3_deriviate_wd3_f_e3_deriviate = np.matmul(err_f_d3_deriviate_wd3, f_e3_deriviate)
        err_f_d2_deriviate_wd3_f_e3_deriviate_h2 = np.matmul(np.transpose(err_f_d3_deriviate_wd3_f_e3_deriviate),
                                                             np.transpose(h2))

        we3 = np.add(we3, eta * err_f_d2_deriviate_wd3_f_e3_deriviate_h2)
        wd3 = np.add(wd3, eta * err_f_d3_deriviate_h3)

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train)) / np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_test[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 3 Encoder
        net_e3 = np.matmul(we3, h2)
        h3 = sigmoid(net_e3)
        h3 = np.reshape(h3, (-1, 1))

        # Layer 3 Decoder
        net_d3 = np.matmul(wd3, h3)
        h2_hat = sigmoid(net_d3)
        h2_hat = np.reshape(h2_hat, (-1, 1))

        output_train.append(h2_hat)

        # Error
        err = y_test[j] - h2_hat
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test)) / np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)

train_loss = np.squeeze(np.array(MSE_train)).mean(axis=1)
test_loss = np.squeeze(np.array(MSE_test)).mean(axis=1)

# plot the train and test loss
plt.plot(range(epochs), train_loss, 'b', label='Train Loss')
plt.plot(range(epochs), test_loss, 'r', label='Test Loss')
plt.title('Train and Test Loss of Autoencoder')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("==============================================================================================================")
print("==============================================================================================================")
print("Train mean mse for each epoch on AE three")
print(np.mean(np.squeeze(np.asarray(MSE_train)), axis=1))
print("==============================================================================================================")
print("Test mean mse for each epoch on AE three")
print(np.mean(np.squeeze(np.asarray(MSE_test)), axis=1))

MSE_train = []
MSE_test = []

# Three MLP Layer global train
for i in range(epochs):

    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_train[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 3 Encoder
        net_e3 = np.matmul(we3, h2)
        h3 = sigmoid(net_e3)
        h3 = np.reshape(h3, (-1, 1))

        # Layer 1 MLP
        net1 = np.matmul(w1, h3)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1, (-1, 1))

        # Layer 2 MLP
        net2 = np.matmul(w2, o1)
        o2 = sigmoid(net2)
        o2 = np.reshape(o2, (-1, 1))

        # Layer 3 MLP
        net3 = np.matmul(w3, o2)
        o3 = net3

        output_train.append(o3[0])

        # Error
        err = y_train[j] - o3[0]
        sqr_err_epoch_train.append(err ** 2)

        # Back propagation
        f1_deriviate = sigmoid_deriviate(net1)
        f2_deriviate = sigmoid_deriviate(net2)
        fe3_deriviate = sigmoid_deriviate(net_e3)
        fe2_deriviate = sigmoid_deriviate(net_e2)
        fe1_deriviate = sigmoid_deriviate(net_e1)

        w3_f2_deriviate = np.matmul(w3, f2_deriviate)

        w3_f2_deriviate_o1 = np.matmul(np.transpose(w3_f2_deriviate), np.transpose(o1))
        w3_f2_deriviate_w2 = np.matmul(w3_f2_deriviate, w2)

        w3_f2_deriviate_w2_f1_deriviate = np.matmul(w3_f2_deriviate_w2, f1_deriviate)

        w3_f2_deriviate_w2_f1_deriviate_h3 = np.matmul(np.transpose(w3_f2_deriviate_w2_f1_deriviate), np.transpose(h3))
        w3_f2_deriviate_w2_f1_deriviate_w1 = np.matmul(w3_f2_deriviate_w2_f1_deriviate, w1)

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate = np.matmul(w3_f2_deriviate_w2_f1_deriviate_w1, fe3_deriviate)

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_h2 = np.matmul(np.transpose(
            w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate), np.transpose(h2)
        )
        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3 = np.matmul(
            w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate, we3
        )

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate = np.matmul(
            w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3, fe2_deriviate
        )

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_h1 = np.matmul(
            np.transpose(w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate), np.transpose(h1)
        )
        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2 = np.matmul(
            w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate, we2
        )

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2_fe1_deriviate = np.matmul(
            w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2, fe1_deriviate
        )

        w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2_fe1_deriviate_x = np.matmul(
            np.transpose(w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2_fe1_deriviate),
            np.reshape(x_train[j], (1, -1))
        )

        we1 = np.add(
            we1, eta * err * w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_we2_fe1_deriviate_x
        )
        we2 = np.add(we2, eta * err * w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_we3_fe2_deriviate_h1)
        we3 = np.add(we3, eta * err * w3_f2_deriviate_w2_f1_deriviate_w1_fe3_deriviate_h2)
        w1 = np.add(w1, eta * err * w3_f2_deriviate_w2_f1_deriviate_h3)
        w2 = np.add(w2, eta * err * w3_f2_deriviate_o1)
        w3 = np.add(w3, eta * err * np.transpose(o2))

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train)) / np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1 Encoder
        net_e1 = np.matmul(we1, x_test[j])
        h1 = sigmoid(net_e1)
        h1 = np.reshape(h1, (-1, 1))

        # Layer 2 Encoder
        net_e2 = np.matmul(we2, h1)
        h2 = sigmoid(net_e2)
        h2 = np.reshape(h2, (-1, 1))

        # Layer 3 Encoder
        net_e3 = np.matmul(we3, h2)
        h3 = sigmoid(net_e3)
        h3 = np.reshape(h3, (-1, 1))

        # Layer 1 MLP
        net1 = np.matmul(w1, h3)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1, (-1, 1))

        # Layer 2 MLP
        net2 = np.matmul(w2, o1)
        o2 = sigmoid(net2)
        o2 = np.reshape(o2, (-1, 1))

        # Layer 3 MLP
        net3 = np.matmul(w3, o2)
        o3 = net3

        output_test.append(o3[0])

        # Error
        err = y_test[j] - o3[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test)) / np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)

    # # Ploy fits
    #
    # # Train
    # m_train, b_train = np.polyfit(y_train, output_train, 1)
    #
    # # Test
    # m_test, b_test = np.polyfit(y_test, output_test, 1)
    #
    # print(m_train, b_train, m_test, b_test)
    #
    # # Plots
    # fig, axs = plt.subplots(3, 2)
    # axs[0, 0].plot(MSE_train, 'b')
    # axs[0, 0].set_title('MSE Train')
    # axs[0, 1].plot(MSE_test, 'r')
    # axs[0, 1].set_title('Mse Test')
    #
    # axs[1, 0].plot(y_train, 'b')
    # axs[1, 0].plot(output_train, 'r')
    # axs[1, 0].set_title('Output Train')
    # axs[1, 1].plot(y_test, 'b')
    # axs[1, 1].plot(output_test, 'r')
    # axs[1, 1].set_title('Output Test')
    #
    # axs[2, 0].plot(y_train, output_train, 'b*')
    # axs[2, 0].plot(y_train, m_train * y_train + b_train, 'r')
    # axs[2, 0].set_title('Regression Train')
    # axs[2, 1].plot(y_test, output_test, 'b*')
    # axs[2, 1].plot(y_test, m_test * y_test + b_test, 'r')
    # axs[2, 1].set_title('Regression Test')
    # if i == (epochs - 1):
    #     plt.savefig('Results.jpg')
    # plt.show()
    # time.sleep(1)
    # plt.close(fig)

    # Ploy fits

    # Train
    m_train, b_train = np.polyfit(y_train, output_train, 1)

    # Test
    m_test, b_test = np.polyfit(y_test, output_test, 1)

    print(m_train, b_train, m_test, b_test)

    # Plots
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(MSE_train, 'b')
axs[0, 0].set_title('MSE Train')
axs[0, 1].plot(MSE_test, 'r')
axs[0, 1].set_title('Mse Test')

axs[1, 0].plot(y_train, 'b')
axs[1, 0].plot(output_train, 'r')
axs[1, 0].set_title('Output Train')
axs[1, 1].plot(y_test, 'b')
axs[1, 1].plot(output_test, 'r')
axs[1, 1].set_title('Output Test')

axs[2, 0].plot(y_train, output_train, 'b*')
axs[2, 0].plot(y_train, m_train * y_train + b_train, 'r')
axs[2, 0].set_title('Regression Train')
axs[2, 1].plot(y_test, output_test, 'b*')
axs[2, 1].plot(y_test, m_test * y_test + b_test, 'r')
axs[2, 1].set_title('Regression Test')
plt.show()
