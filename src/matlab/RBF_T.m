clc;close all;clear;

%% Read and Normalize Dataset
data=xlsread('Tehran(Geophysic)_T.xlsx');

n=size(data,1);
m=size(data,2);

data_min = min(data);
data_max = max(data);

% Normalize data
for i = 1:n
    for j = 1:m
        data(i, j) = (data(i, j) - data_min) / (data_max - data_min);
    end
end

%% Initialize Parameters
train_rate=0.75;
test_rate=0.25;

num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;
        
data_train=data(1:num_of_train,:);
data_test=data(num_of_train+1:n,:);

eta_w=0.03;
eta_m=0.03;
eta_s=0.03;
epoch=400;

n0=m-1;  % n0
n1=4;
n2=1;

lowerb=-1;
upperb=1;

%% Initialize Perceptron weigths
mean = unifrnd(lowerb,upperb, [n1 n0]);
sigma = unifrnd(lowerb,upperb, [n1 n2]);

w=unifrnd(lowerb,upperb,[n2 n1]);

error_train=zeros(num_of_train,1);
error_test=zeros(num_of_test,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(epoch,1);
mse_test=zeros(epoch,1);


%% 2 Layer Perceptron
% Train
for i=1:epoch
  for j=1:num_of_train
      
    input=data_train(j,1:m-1);
    target=data_train(j,m);

    % Layer 1
    net1=sum((input - mean).^2, 2);  % n1*1
    o1=exp(-0.5 * net1./(sigma.^2));  % n1*1

    % Layer 2
    net2=w*o1;  % n2*1
    o2=net2;  % n2*1

    % Predicted  Train Output
    output_train(j,1)=o2;

    % Calc Error Train
    e=target-o2;  % 1*1
    error_train(j,1)=e;

    % Back Propagation
    delta_mean = eta_m*e*-1*w'*((input - mean)'*(o1./(sigma.^2)))';  % n1*n0 = 1*1 * 1*1 * n1*1 * (n0*n1 * (n1*1/n1*1))'
    delta_sigma=eta_s*e*-1*w'.*sum((input - mean).^2, 2).*(1./(sigma.^3)).*o1;  % n1*1 = 1*1 * 1*1 * n1*1 .* n1*1.* n1*1 .* n1*1
    mean=mean-delta_mean;
    sigma=sigma-delta_sigma;
    w=w-eta_w*e*-1*1*o1';  % 1*n1 = 1*n1 - 1*1 * 1*1 * 1*n1
           
  end
  
  % Mean Square Train Error
  mse_train(i,1)=mse(error_train);
  
  % Test
  for j=1:num_of_test
      
    input=data_test(j,1:m-1);
    target=data_test(j,m);

    % Layer 1
    net1=sum((input - mean).^2, 2);  % n1*1
    o1=exp(-0.5 * net1./(sigma.^2));  % n1*1

    % Layer 2
    net2=w*o1;  % n2*1
    o2=net2;  % n2*1

    % Predicted Output
    output_test(j,1)=o2;

    % Calc Error
    e=target-o2;  % 1*1
    error_test(j,1)=e;     
      
  end 
  
  % Mean Square Test Error
  mse_test(i,1)=mse(error_test);
  
  %% Find Regression
  [m_train ,b_train]=polyfit(data_train(:,m),output_train(:,1),1);
  [y_fit_train,~] = polyval(m_train,data_train(:,m),b_train);
  [m_test ,b_test]=polyfit(data_test(:,m),output_test(:,1),1);
  [y_fit_test,~] = polyval(m_test,data_test(:,m),b_test);
  
  %% Plot Results
  figure(1);
  subplot(2,3,1),plot(data_train(:,m),'-r');
  hold on;
  subplot(2,3,1),plot(output_train,'-b');
  title('Output Train')
  hold off;
  
  subplot(2,3,2),semilogy(mse_train(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(data_train(:,m),output_train(:,1),'b*');hold on;
  plot(data_train(:,m),y_fit_train,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(data_test(:,m),'-r');
  hold on;
  subplot(2,3,4),plot(output_test,'-b');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(data_test(:,m),output_test(:,1),'b*');hold on;
  plot(data_test(:,m),y_fit_test,'r-');
  title('Regression Test')
  hold off;
  
  pause(0.01);
end