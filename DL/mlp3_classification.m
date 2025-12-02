

close all;
clear all;

%% Read and Normalize Dataset of train
data1 = readmatrix('train(3_class_condition).xlsx');
n1 = size(data1,1);
m1 = size(data1,2);
input_num = m1-1;
data_min1 = min(data1);
data_max1 = max(data1);clc;
for i = 1:n1
    for j = 1:m1
        data1(i,j) = (data1(i,j)-data_min1(1,j))/(data_max1(1,j)-data_min1(1,j));
    end
end

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
%% Parameters of network
eta_p = 0.01;
max_epoch_p = 60;
n0_neurons = input_num;
n1_neurons = 25;
n2_neurons = 13;
n3_neurons = 3;
lowerb=-1;
upperb=1;
%% Train and Test Data
num_of_train=n1;
num_of_test=n2;
%% weigths of MLP
w1=unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
net1=zeros(n1_neurons,1);
o1=zeros(n1_neurons,1);

w2=unifrnd(lowerb,upperb,[n2_neurons  n1_neurons]);
net2=zeros(n2_neurons ,1);
o2=zeros(n2_neurons ,1);

w3=unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);
net3=zeros(n3_neurons,1);
o3=zeros(n3_neurons,1);

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
      input=data2(j,1:m2-1);
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
%% 3 Layer MLP
% Train part
y_pred1 = zeros(num_of_train, 1);
for i=1:max_epoch_p
    for j=1:num_of_train

        input=data1(j,1:m1-1);
        target_train = target1(j,:)';

        % Feed-Forward

        % Layer 1
        net1=w1*input'; 
        o1=logsig(net1);  

        % Layer 2
        net2=w2*o1; 
        o2=logsig(net2);  

        % Layer 3
        net3=w3*o2;
        o3=softmax(net3);
        e=o3 - target_train;
        loss = -sum(target_train.*log(o3)); % cross-entropy loss

        % Compute loss on training data
        loss_values_train(i) = loss_values_train(i) + loss/num_of_train;

      % Back Propagation
      f1_driviate=diag(o1.*(1-o1));
      f2_driviate=diag(o2.*(1-o2));
      
      w1=w1-eta_p*(e'*(w3*f2_driviate*w2*f1_driviate))'*input;
      w2=w2-eta_p*(e'*(w3*f2_driviate))'*o1';
      w3=w3-eta_p*e*o2';

% Determine predicted label
     [~, idx] = max(o3);
     if idx == 1
         y_pred1(j) = 0;
     elseif idx == 2
         y_pred1(j) = 0.5;
     else
         y_pred1(j) = 1;
     end
    end
y_pred2 = zeros(num_of_test, 1);
% Test part
  for j=1:num_of_test
      
      input=data2(j,1:m2-1);
      target_test=target2(j,:)';
      
% 
      % Feed-Forward
      
      % Layer 1
      net1=w1*input'; 
      o1=logsig(net1);
      
      % Layer 2
      net2=w2*o1;
      o2=logsig(net2); 
      
      % Layer 3
       net3=w3*o2;
       o3=softmax(net3);
       e=o3 - target_test;
       loss1 = -sum(target_test.*log(o3));

      % Compute the loss
      loss_values_test(i) = loss_values_test(i) + loss1/num_of_test;

% Determine predicted label
    [~, idx] = max(o3);
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
  heatmap(labels, labels, c1);
  title('confusion matrix of train ');

  c2 = confusionmat(data2(:, m2), y_pred2);
  subplot(2,2,4),c2;
  labels = {'no stress','time pressure','interruption'};
  heatmap(labels, labels, c2);
  title('confusion matrix of test ');
end