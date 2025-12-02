%MLP 2 layers
clc;
close all;
clear all;
%% Read Data
x=xlsread('Book2');
N = size(x,1);
D=15;
L=4;
[Data,Target]=Make_Data(x,N,D,L);
data=[Data,Target];
% data = Normal_func(data,0,+1);
for i=3:7733
    for j=1:16
        if isnan(data(i,j))
            data(i,j)=(data(i-2,j)+data(i-1,j))/2;
        end
    end
end

%% Normalize Dataset
data_min = min(data);
data_max = max(data);
n=size(data,1);
m=size(data,2);
for i = 1:n
    for j = 1:m
        data(i, j) = (data(i, j) - data_min(1,j)) / (data_max(1,j) - data_min(1,j));
    end
end
%% initialize parametrs
train_rate=0.2;

num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;

data_train=data(1:num_of_train,:);
data_test=data(1:num_of_test,:);

eta=0.001;
max_epoch=100;

n1=m-1;
n2=7;
n3=1;

lowerb=-1;
upperb=1;

w1=unifrnd(lowerb,upperb,[n2 n1]);
net1=zeros(n2,1);
o1=zeros(n2,1);

w2=unifrnd(lowerb,upperb,[n3 n2]);
net2=zeros(n3,1);
o2=zeros(n3,1);

error_train=zeros(num_of_train,1);
error_test=zeros(num_of_test,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(max_epoch,1);
mse_test=zeros(max_epoch,1);
%% Train & Test process
for i=1:max_epoch
  for j=1:num_of_train
      input=data_train(j,1:m-1);
      target=data_train(j,m);
      net1=w1*input';
      o1=logsig(net1);
      net2=w2*o1;
      o2=net2;
      
      e=target-o2;

      output_train(j,1)=o2;
      error_train(j,1)=e;
      
       A=diag(o1.*(1-o1));

      w1=w1-eta*e*-1*1*(w2*A)'*input;
      w2=w2-eta*e*-1*1*o1';
           
  end
  
  mse_train(i,1)=mse(error_train);
  
  for j=1:num_of_test
      
      input=data_test(j,1:m-1);
      target=data_test(j,m);
      net1=w1*input';
      o1=logsig(net1);
      net2=w2*o1;
      o2=net2;
      
      e=target-o2;
      output_test(j,1)=o2;
      error_test(j,1)=e;
      
      
  end 
  
  mse_test(i,1)=mse(error_test);
  
  %% Find Regression
  [m_train ,b_train]=polyfit(data_train(:,m),output_train(:,1),1);
  [y_fit_train,~] = polyval(m_train,data_train(:,m),b_train);
  [m_test ,b_test]=polyfit(data_test(:,m),output_test(:,1),1);
  [y_fit_test,~] = polyval(m_test,data_test(:,m),b_test);
  %% Draw results
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
  
  pause(0.001);
end
fprintf('mse train = %1.16g, mse test = %1.16g \n', mse_train(max_epoch,1), mse_test(max_epoch,1))






