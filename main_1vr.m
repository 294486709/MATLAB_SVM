function main_1_r()
clear ; close all; clc
load MNIST_data.mat;
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
Nsample = size(train_samples,1);
label0 =  zeros(Nsample,1);
label1 =  zeros(Nsample,1);
label2 =  zeros(Nsample,1);
label3 =  zeros(Nsample,1);
label4 =  zeros(Nsample,1);
label5 =  zeros(Nsample,1);
label6 =  zeros(Nsample,1);
label7 =  zeros(Nsample,1);
label8 =  zeros(Nsample,1);
label9 =  zeros(Nsample,1);
for i = 1:Nsample
    if train_samples_labels(i) == 0
        label0(i)=1;
    elseif train_samples_labels(i) == 1
        label1(i)=1;
    elseif train_samples_labels(i) == 2
        label2(i)=1;
    elseif train_samples_labels(i) == 3
        label3(i)=1;
    elseif train_samples_labels(i) == 4
        label4(i)=1;
    elseif train_samples_labels(i) == 5
        label5(i)=1;
    elseif train_samples_labels(i) == 6
        label6(i)=1;
    elseif train_samples_labels(i) == 7
        label7(i)=1;
    elseif train_samples_labels(i) == 8
        label8(i)=1;
    else
        label9(i)=1;
    end
end
C = 0.1;
sigma = 0.03; 
model_0 = svmTrain(train_samples, label0, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("1")
model_1 = svmTrain(train_samples, label1, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("2")
model_2 = svmTrain(train_samples, label2, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("3")
model_3 = svmTrain(train_samples, label3, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("4")
model_4 = svmTrain(train_samples, label4, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("5")
model_5 = svmTrain(train_samples, label5, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("6")
model_6 = svmTrain(train_samples, label6, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("7")
model_7 = svmTrain(train_samples, label7, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("8")
model_8 = svmTrain(train_samples, label8, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("9")
model_9 = svmTrain(train_samples, label9, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
disp("10")
predictions = svmPredict(model_0, test_samples);
Ntest = size(test_samples,1);
correctcount=0;
for i=1:Ntest
    if test_samples_labels(i) == 0
        current_m = model_0;
    elseif test_samples_labels(i) == 1
                current_m = model_1;
    elseif test_samples_labels(i) == 2
                current_m = model_2;
    elseif test_samples_labels(i) == 3
                current_m = model_3;
    elseif test_samples_labels(i) == 4
                current_m = model_4;
    elseif test_samples_labels(i) == 5
                current_m = model_5;
    elseif test_samples_labels(i) == 6
                current_m = model_6;
    elseif test_samples_labels(i) == 7
                current_m = model_7;
    elseif test_samples_labels(i) == 8
                current_m = model_8;
    elseif test_samples_labels(i) == 9
                current_m = model_9;
    end
    pred = svmPredict(current_m,test_samples(i,:));
    if pred == 1
        correctcount = correctcount + 1;
    end
end

fprintf("Accuracy is:")
disp(correctcount/Ntest)
end
