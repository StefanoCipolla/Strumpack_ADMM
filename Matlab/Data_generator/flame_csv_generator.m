clear all
clc
close all
% Dataset
filename = "2moons";
% Loading Training Data 
A  = readmatrix(filename  + ".txt");
A(A==2.0)=-1;
p=randperm(size(A,1));

% 100 points to test
test_data  = A(p(1:100),1:2);
test_label = A(p(1:100),3);
% The remaining are train_data 
train_data  = A(p(101:end),1:2);
train_label = A(p(101:end),3);

data1=train_data(train_label==1,:);
data2=train_data(train_label==-1,:);

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)


writematrix(train_data,'2M_train.csv');
writematrix(train_label,'2M_train_label.csv');
writematrix(test_data,'2M_test.csv');
writematrix(test_label,'2M_test_label.csv');
