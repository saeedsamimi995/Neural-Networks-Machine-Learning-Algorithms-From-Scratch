clc;
close all;
clear all;

%% Read and Normalize Dataset of train
data1 = readmatrix('train(3_class_condition).xlsx');
n1 = size(data1,1);
m1 = size(data1,2);
input_num = m1-1;
data_min1 = min(data1);
data_max1 = max(data1);

for i = 1:n1
    for j = 1:m1
        data1(i,j) = (data1(i,j)-data_min1(1,j))/(data_max1(1,j)-data_min1(1,j));
    end
end
x = data1(:, 1:input_num);
%% Read and Normalize Dataset of test
data2 = readmatrix('test(3_class_condition).xlsx');
n2 = size(data2,1);
m2 = size(data2,2);
input_num = m2-1;
data_min2 = min(data2);
data_max2 = max(data2);

for i = 1:n2
    for j = 1:m2
        data2(i,j) = (data2(i,j)-data_min2(1,j))/(data_max2(1,j)-data_min2(1,j));
    end
end
%% Initialize Parameters
eta_e1 = 0.01;
eta_e2 = 0.02;
eta_e3 = 0.03;
eta_p = 0.01;
epochs_ae = 30;
max_epoch_p = 60;
drop_out_rate = 0.4;

n0_neurons = input_num;
n1_neurons = 100;
n2_neurons = 50;
n3_neurons = 25;

l1_neurons=8;
l2_neurons=3;

lowerb=-1;
upperb=1;

%% Initialize Autoencoder weigths
w_e1 = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
w_e2 = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
w_e3 = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);

w_d1 = unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
w_d2 = unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
w_d3 = unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);

net_e1=zeros(n1_neurons,1);
h1=zeros(n1_neurons,1);
net_d1=zeros(n0_neurons,1);
x_hat=zeros(n0_neurons,1);

net_e2=zeros(n2_neurons,1);
h2=zeros(n2_neurons,1);
net_d2=zeros(n1_neurons,1);
h1_hat=zeros(n1_neurons,1);

net_e3=zeros(n3_neurons,1);
h3=zeros(n3_neurons,1);
net_d3=zeros(n2_neurons,1);
h2_hat=zeros(n2_neurons,1);

%% Encoder 1 Local Train
for i=1:epochs_ae
    for j=1:n1

        % drop out
        drop_out_num1 = round(n1_neurons * drop_out_rate);
        randomlist_e1 = randi(n1_neurons, drop_out_num1, 1);
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
        h1 = logsig(net_e1);  % n1*1
        for k=1:size(randomlist_e1,1)
            h1(randomlist_e1(k,1), :) = 0;
        end

        % Decoder1
        net_d1 = w_d1 * h1;  % n0*1
        x_hat = logsig(net_d1);  % n0*1

        % Error
        err = x(j,:) - x_hat';  % 1*n0

        % Back propagation
        w_e1_hold = w_e1;
        f_driviate_d = diag(x_hat.*(1-x_hat));  % n0*n0
        f_driviate_e = diag(h1.*(1-h1));  % n1*n1
        delta_w_d1 = (eta_e1 * h1 * err * f_driviate_d)';  % n0*n1 = (1*1 * n1*1 * 1*n0 * n0*n0)'
        delta_w_e1 = (eta_e1 * x(j,:)' * err * f_driviate_d * w_d1 * f_driviate_e)';  % n1*n0 = (1*1 * n0*1 * 1*n0 * n0*n0 * n0*n1 * n1*n1)'
        w_d1 = w_d1 + delta_w_d1;  % n0*n1
        w_e1 = w_e1 + delta_w_e1;  % n1*n0
        for k=1:size(randomlist_e1,1)
            w_e1(randomlist_e1(k,1), :) = w_e1_hold(randomlist_e1(k,1), :);  % Drop Out Weights Stay Unchanged
        end
    end
end

%% Encoder 2 Local Train
for i=1:epochs_ae
    for j=1:n1

        % Drop out
        drop_out_num2 = round(n2_neurons * drop_out_rate);
        randomlist_e2 = randi(n2_neurons, drop_out_num2, 1);

        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
        h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
        h2 = logsig(net_e2);  % n2*1
        for k=1:size(randomlist_e2,1)
            h2(randomlist_e2(k,1), :) = 0;
        end

        % Decoder2
        net_d2 = w_d2 * h2;  % n1*1
        h1_hat = logsig(net_d2);  % n1*1

        % Error
        err = (h1 - h1_hat)';  % 1*n1

        % Back propagation
        w_e2_hold = w_e2;
        f_driviate_d = diag(h1_hat.*(1-h1_hat));  % n1*n1
        f_driviate_e = diag(h2.*(1-h2));  % n2*n2
        delta_w_d2 = (eta_e2 * h2 * err * f_driviate_d)';  % n1*n2 = (1*1 * n2*1 * 1*n1 * n1*n1)'
        delta_w_e2 = (eta_e2 * h1 * err * f_driviate_d * w_d2 * f_driviate_e)';  % n2*n1 = (1*1 * n1*1 * 1*n1 * n1*n1 * n1*n2 * n2*n2)'
        w_d2 = w_d2 + delta_w_d2;  % n1*n2
        w_e2 = w_e2 + delta_w_e2;  % n2*n1
        for k=1:size(randomlist_e2,1)
            w_e2(randomlist_e2(k,1), :) = w_e2_hold(randomlist_e2(k,1), :);  % Drop Out Weights Stay Unchanged
        end
    end
end
%% Encoder 3 Local Train
for i=1:epochs_ae
    for j=1:n1

        % Drop out
        drop_out_num3 = round(n3_neurons * drop_out_rate);
        randomlist_e3 = randi(n3_neurons, drop_out_num3, 1);
        
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
        h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
        h2 = logsig(net_e2);  % n2*1
       
        % Encoder3
        net_e2 = w_e3* h2;  % n3*1
        h3 = logsig(net_e2);  % n3*1
        for k=1:size(randomlist_e3,1)
            h3(randomlist_e3(k,1), :) = 0;
        end

        % Decoder3
        net_d3 = w_d3 * h3;  % n1*1
        h2_hat = logsig(net_d3);  % n1*1

        % Error
        err = (h2 - h2_hat)';  % 1*n1

        % Back propagation
        w_e3_hold = w_e3;
        f_driviate_d = diag(h2_hat.*(1-h2_hat));  % n2*n2
        f_driviate_e = diag(h3.*(1-h3));  % n3*n3
        delta_w_d3 = (eta_e3 * h3 * err * f_driviate_d)';  % n2*n3 = (1*1 * n3*1 * 1*n2 * n2*n2)'
        delta_w_e3 = (eta_e3 * h2 * err * f_driviate_d * w_d3 * f_driviate_e)';  % n3*n2 = (1*1 * n2*1 * 1*n2 * n2*n2 * n2*n3 * n3*n3)'
        w_d3 = w_d3 + delta_w_d3;  % n2*n3
        w_e3 = w_e3 + delta_w_e3;  % n3*n2
        for k=1:size(randomlist_e3,1)
            w_e3(randomlist_e3(k,1), :) = w_e3_hold(randomlist_e3(k,1), :);  % Drop Out Weights Stay Unchanged
        end
    end
end

%% Train and Test Data
num_of_train=n1;
num_of_test=n2;
%% Initialize Perceptron weigths
w1=unifrnd(lowerb,upperb,[l1_neurons n3_neurons]);
net1=zeros(l1_neurons,1);
o1=zeros(l1_neurons,1);

w2=unifrnd(lowerb,upperb,[l2_neurons l1_neurons]);
net2=zeros(l2_neurons,1);
o2=zeros(l2_neurons,1);

target1 = zeros(3 , num_of_train);
target2 = zeros(3 , num_of_test);

%% give lable for data
for j = 1:num_of_train
      input=data1(j,1:m1-1);
      target = data1(j,m1);
      if data1(j,m1)==0
          target1(:,j) = [1 0 0];
      elseif data1(j,m1)==0.5
              target1(:,j) = [0 1 0];
      else 
              target1(:,j) = [0 0 1];
      end
end
target1= target1';


for j = 1:num_of_test
      target=data2(j,m2);
      if data2(j,m2)==0
          target2(:,j) = [1 0 0];
      elseif data2(j,m2)==0.5
              target2(:,j) = [0 1 0];
      else 
              target2(:,j) = [0 0 1];
      end
end
target2= target2';

loss_values_train = zeros(max_epoch_p, 1);
loss_values_test = zeros(max_epoch_p, 1);
%% 2 Layer Perceptron
% Train
y_pred1 = zeros(num_of_train, 1);

for i=1:max_epoch_p
  for j=1:num_of_train
      
      input=data1(j,1:m1-1);
      target_train = target1(j,:)';

      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  % n1*1
      h1 = logsig(net_e1);  % n1*1

      % Encoder2
      net_e2 = w_e2* h1;  % n2*1
      h2 = logsig(net_e2);  % n2*1

      % Encoder3
      net_e3 = w_e3* h2;  % n2*1
      h3 = logsig(net_e3);  % n2*1

      % Layer 1
      net1=w1*h3;  % l1*1
      o1=logsig(net1);  % l1*1
      
      % Layer 2
      net2=w2*o1;  % l2*1
      o2=softmax(net2); 
      
      % Calc Error
      e=o2 - target_train;  % 1*1

      loss = -sum(target_train.*log(o2)); % cross-entropy loss

      % Compute loss on training data
      loss_values_train(i) = loss_values_train(i) + loss/num_of_train;

      
      % Back Propagation
      f_driviate=diag(o1.*(1-o1)); 
      w1=w1-eta_p*(e'*(w2*f_driviate))'*h3';  
      w2=w2-eta_p*e*o1';
 % Determine predicted label
      [~, idx] = max(o2);
      if idx == 1
          y_pred1(j) = 0;
      elseif idx == 2
          y_pred1(j) = 0.5;
      else
          y_pred1(j) = 1;
      end
  end
  
y_pred2 = zeros(num_of_test, 1);
  
  % Test
  for j=1:num_of_test
      
      input=data2(j,1:m2-1);
      target_test=target2(j,:)';
      
      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  
      h1 = logsig(net_e1);  

      % Encoder2
      net_e2 = w_e2* h1;  
      h2 = logsig(net_e2);  
      
      % Encoder3
      net_e3 = w_e3* h2;  
      h3 = logsig(net_e3);  
      
      % Layer 1
      net1=w1*h3;  
      o1=logsig(net1);  
      
      % Layer 2
      net2=w2*o1;  
      o2=softmax(net2);  
      
      e=o2 - target_test;
      loss1 = -sum(target_test.*log(o2));

      % Compute the loss
      loss_values_test(i) = loss_values_test(i) + loss1/num_of_test;
 % Determine predicted label
    [~, idx] = max(o2);
    if idx == 1
        y_pred2(j) = 0;
    elseif idx == 2
        y_pred2(j) = 0.5;
    else
        y_pred2(j) = 1;
    end
end
figure(1)
  subplot(2,2,1),plot(loss_values_train(1:i,1),'-r');
  title('error of train')
  hold off;

  subplot(2,2,3),plot(loss_values_test(1:i,1),'-r');
  title('error of test')
  hold off;

  c1 = confusionmat(data1(:, m1), y_pred1);
  subplot(2,2,2),c1;
  labels = {'no stress','time pressure','interruption'};
  heatmap(labels, labels, c1 , 'colormap' , cool);
  title('confusion matrix of train ');

  c2 = confusionmat(data2(:, m2), y_pred2);
  subplot(2,2,4),c2;
  labels = {'no stress','time pressure','interruption'};
  heatmap(labels, labels, c2 , 'colormap' , cool);
  title('confusion matrix of test ');
end