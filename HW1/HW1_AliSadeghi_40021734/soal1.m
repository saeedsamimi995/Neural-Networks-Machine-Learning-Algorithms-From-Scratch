clc
clear all

%% Read and Normalize Dataset
data = xlsread('Bias_correction_ucl(NaNfree).xlsx');
n = size(data,1);
m = size(data,2);
input_num = m-1;
data_min = min(data);
data_max = max(data);

for i = 1:n
    for j = 1:m
        data(i,j) = (data(i,j)-data_min(1,j))/(data_max(1,j)-data_min(1,j));
    end
end
x = data(:,1:input_num);

%% Parameters of network
train_rate=0.7;
eta_p = 0.01;
max_epoch_p = 60;
n0_neurons = input_num;
n1_neurons = 5;
n2_neurons = 1;
lowerb=-1;
upperb=1;

%% Train and Test Data
num_of_train=round(train_rate*n);
data_train=data(1:num_of_train,:);
num_of_test=n-num_of_train;
data_test=data(num_of_train+1:n,:);

%% weigths of MLP
w1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
net1=zeros(n1_neurons,1);
o1=zeros(n1_neurons,1);

w2=unifrnd(lowerb,upperb,[n2_neurons  n1_neurons]);
net2=zeros(n2_neurons ,1);
o2=zeros(n2_neurons ,1);

error_train=zeros(num_of_train,1);
output_train=zeros(num_of_train,1);
mse_train=zeros(max_epoch_p,1);
error_test=zeros(num_of_test,1);
output_test=zeros(num_of_test,1);
mse_test=zeros(max_epoch_p,1);

%% 2 Layer MLP
% Train part
for i=1:max_epoch_p
  for j=1:num_of_train
      
      input=data_train(j,1:m-1);
      target=data_train(j,m);

      % Feed-Forward

      % Layer 1
      net1=w1*input'; 
      o1=logsig(net1);  
      
      % Layer 2
      net2=w2*o1; 
      o2=net2;  
      
      output_train(j,1)=o2;
      
      e=target-o2;  
      error_train(j,1)=e;
      
      % Back Propagation
      f_driviate=diag(o1.*(1-o1));
      w1=w1-eta_p*e*-1*1*(w2*f_driviate)'*input;
      w2=w2-eta_p*e*-1*1*o1'; 
  end
  
  % MSE Train
  mse_train(i,1)=mse(error_train);
  
  % Test part
  for j=1:num_of_test
      
      input=data_test(j,1:m-1);
      target=data_test(j,m);
      
      % Feed-Forward
      
      % Layer 1
      net1=w1*input'; 
      o1=logsig(net1);
      
      % Layer 2
      net2=w2*o1;
      o2=net2; 
      
      output_test(j,1)=o2;
      
      e=target-o2; 
      error_test(j,1)=e;
  end 
  
  % MSE Test
  mse_test(i,1)=mse(error_test);
  
  %% Regression
  [m_train ,b_train]=polyfit(data_train(:,m),output_train(:,1),1);
  [y_fit_train,~] = polyval(m_train,data_train(:,m),b_train);
  [m_test ,b_test]=polyfit(data_test(:,m),output_test(:,1),1);
  [y_fit_test,~] = polyval(m_test,data_test(:,m),b_test);
  
  %% Plots of results
  figure(1);
  subplot(2,3,1),plot(data_train(:,m),'-r');
  hold on;
  subplot(2,3,1),plot(output_train,'-g');
  title('Output Train')
  hold off;
  
  subplot(2,3,2),semilogy(mse_train(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(data_train(:,m),output_train(:,1),'g*');hold on;
  plot(data_train(:,m),y_fit_train,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(data_test(:,m),'-r');
  hold on;
  subplot(2,3,4),plot(output_test,'-g');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(data_test(:,m),output_test(:,1),'g*');hold on;
  plot(data_test(:,m),y_fit_test,'r-');
  title('Regression Test')
  hold off;
  
  pause(0.001);
end

fprintf('mse train = %1.16g, mse test = %1.16g \n', mse_train(max_epoch_p,1), mse_test(max_epoch_p,1))