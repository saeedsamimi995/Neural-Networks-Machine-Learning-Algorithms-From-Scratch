# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import timeit

# =============================================================================
# Definition and derivation of gaussian
# =============================================================================
# mean: matrix, sigma: vector
def gaussian(x, mean, sigma):
    sumation = np.sum(np.power((x - mean), 2), 1)
    return np.exp(-0.5 * (1/np.power(np.transpose(sigma), 2)) * sumation)

def gaussian_deriviate_mean(x, mean, sigma, o1):
    sumation = x - mean
    return (1/np.power(sigma, 2)) * sumation * np.transpose(o1)

def gaussian_deriviate_sigma(x, mean, sigma, o1):
    sumation = np.sum(np.power((x - mean), 2), 1)
    sumation = np.reshape(sumation, (-1, 1))
    return (1/np.power(sigma, 3)) * sumation * np.transpose(o1)

# =============================================================================
# Read, show and normalize data
# ============================================================================= 
data = pd.read_excel('./data/Tehran_Geophysic_T_P.xlsx')
data_T = np.array(pd.DataFrame(data, columns=['T']))
data_P = np.array(pd.DataFrame(data, columns=['P']))
#
plt.figure(figsize=(20,5))
plt.plot(data_P)
plt.xlabel('Hour', fontweight ='bold')
plt.ylabel('temperature', fontweight ='bold')
plt.show()
#
hold = []
for t in range(4, len(data_P)):
    d = []
    d.append(data_P[t-4].item())
    d.append(data_P[t-3].item())
    d.append(data_P[t-2].item())
    d.append(data_P[t-1].item())
    d.append(data_P[t].item())
    hold.append(d)
data = np.array(hold)
#
min = np.min(data)
max = np.max(data)
for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data[i,j] = (data[i,j] - min) / (max - min)
        
# =============================================================================
# Define train_set - validation_set - test_set
# =============================================================================
split_ratio_train = 0.75
split_ratio_validation = 0.25

split_line_number = int(np.shape(data)[0] * split_ratio_train)
x_train = data[:split_line_number, :4]
y_train = data[:split_line_number, 4]

x_test = data[split_line_number:, :4]
y_test = data[split_line_number:, 4]

# =============================================================================
# Define MLP
# =============================================================================
input_dimension = np.shape(x_train)[1]
l1_neurons = 4
l2_neurons = 1

#np.random.seed(20)
mean = np.random.uniform(low=-1, high=1, size=(l1_neurons, input_dimension))
sigma = np.random.uniform(low=-1, high=1, size=(l1_neurons, l2_neurons))
w = np.random.uniform(low=-1, high=1, size=(l1_neurons, l2_neurons))

# =============================================================================
# Training
# =============================================================================
lrw = 0.05
lrm = 0.05
lrs = 0.05
epochs = 400

MSE_train = []
MSE_test = []

def Train(w, mean, sigma):
    output_train = []
    sqr_err_epoch_train = []
    
    for i in range(np.shape(x_train)[0]):
        x = np.reshape(x_train[i], (1,-1)) # x: (1, 4)
        
        # Feed-Forward
        # Layer 1
        o1 = gaussian(x, mean, sigma) # o1: (1, l1_neurons)
        # Layer 2
        o2 = np.matmul(o1, w) # o2: (1, 1)

        output_train.append(o2[0])

        # Error
        err = y_train[i] - o2[0]
        sqr_err_epoch_train.append(err**2)

        # Back propagation
        mean = np.subtract(mean, (lrm * err * -1 * w * gaussian_deriviate_mean(x, mean, sigma, o1)))
        #
        sigma = np.subtract(sigma, (lrs * err * -1 * w * gaussian_deriviate_sigma(x, mean, sigma, o1)))
        #
        w = np.subtract(w, (lrw * err * -1 * np.transpose(o1)))
        
    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train[0])
    return output_train, w, mean, sigma

def Test(w, mean, sigma):
    sqr_err_epoch_test = []
    output_test = []
    
    for i in range(np.shape(x_test)[0]):
        x = np.reshape(x_test[i], (1,-1))
        # Feed-Forward
        # Layer 1
        o1 = gaussian(x, mean, sigma)
        # Layer 2
        o2 = np.matmul(o1, w)
        
        output_test.append(o2[0])

        # Error
        err = y_test[i] - o2[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test[0])
    return output_test

def Plot_results(output_train, 
                 output_test, 
                 m_train, 
                 b_train,
                 m_test,
                 b_test):
    # Plots
    fig = plt.figure()
    fig.set_size_inches(25, 15)
    
    axs = fig.add_subplot(3, 2, 1)
    axs.plot(MSE_train, 'b')
    axs.set_title('MSE Train')
    axs = fig.add_subplot(3, 2, 2)
    axs.plot(MSE_test, 'r')
    axs.set_title('Mse Test')
    #
    axs = fig.add_subplot(3, 2, 3)
    axs.plot(y_train, 'b')
    axs.plot(output_train, 'r')
    axs.set_title('Output Train')
    axs = fig.add_subplot(3, 2, 4)
    axs.plot(y_test, 'b')
    axs.plot(output_test, 'r')
    axs.set_title('Output Test')
    #
    axs = fig.add_subplot(3, 2, 5)
    axs.plot(y_train, output_train, 'b*')
    axs.plot(y_train, m_train*y_train+b_train,'r')
    axs.set_title('Regression Train')
    axs = fig.add_subplot(3, 2, 6)
    axs.plot(y_test, output_test, 'b*')
    axs.plot(y_test, m_test*y_test+b_test,'r')
    axs.set_title('Regression Test')
    plt.show()
    time.sleep(1)
    plt.close(fig)
    
print('==> Start Training ...')
for epoch in range(epochs):    
    start = timeit.default_timer()
    
    if epoch % 50 == 0:
        lrw = 0.75 * lrw
        lrm = 0.75 * lrm
        lrs = 0.75 * lrs
    
    output_train, w, mean, sigma = Train(w, mean, sigma)
    m_train , b_train = np.polyfit(y_train, output_train, 1)    
    output_test = Test(w, mean, sigma)
    m_test , b_test = np.polyfit(y_test, output_test, 1)
    
    Plot_results(output_train, 
                 output_test, 
                 m_train, 
                 b_train,
                 m_test,
                 b_test)

    stop = timeit.default_timer()
    print('Epoch: {} \t, time: {:.3f}'.format(epoch+1, stop-start))
    print('MSE_train: {:.4f} \t, MSE_test: {:.4f}'.format(MSE_train[epoch], MSE_test[epoch]))
    print(m_train, b_train, m_test, b_test)
print('==> End of training ...')
