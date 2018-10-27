function main_1v1()
clear ; close all; clc
load MNIST_data.mat;
Nsample = size(train_samples,1);
sample0 = [];
sample1 = [];
sample2 = [];
sample3 = [];
sample4 = [];
sample5 = [];
sample6 = [];
sample7 = [];
sample8 = [];
sample9 = [];
for i = 1:Nsample
    if train_samples_labels(i) == 0
        temp = size(sample0,1);
        sample0(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 1
        temp = size(sample1,1);
        sample1(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 2
        temp = size(sample2,1);
        sample2(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 3
        temp = size(sample3,1);
        sample3(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 4
        temp = size(sample4,1);
        sample4(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 5
        temp = size(sample5,1);
        sample5(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 6
        temp = size(sample6,1);
        sample6(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 7
        temp = size(sample7,1);
        sample7(temp+1,:)=train_samples(i,:);
    elseif train_samples_labels(i) == 8
        temp = size(sample8,1);
        sample8(temp+1,:)=train_samples(i,:);
    else
        temp = size(sample9,1);
        sample9(temp+1,:)=train_samples(i,:);
    end
end
s=struct('sam',{});
s(1).sam=sample1;
s(2).sam=sample2;
s(3).sam=sample3;
s(4).sam=sample4;
s(5).sam=sample5;
s(6).sam=sample6;
s(7).sam=sample7;
s(8).sam=sample8;
s(9).sam=sample9;
s(10).sam=sample0;
model = struct('i',{},'j',{},'model',{});
res = struct('vote',{});
C = 1;
sigma = 0.04;
cc=1;
for i = 1:10
    for j = 1:10
        if i <= j
            break;
        end
        training = [s(i).sam;s(j).sam];
        training_label = [zeros(size(s(i).sam,1),1);ones(size(s(j).sam,1),1)];
        model(cc).i = i;
        model(cc).j = j;
        model(cc).model = svmTrain(training, training_label, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        cc = cc + 1;
        disp(cc-1)
    end
end
Ntest = size(test_samples,1);
numbercount=1;
result = zeros(Ntest,1);
for i=1:Ntest
    res(1).vote=0;
    res(2).vote=0;
    res(3).vote=0;
    res(4).vote=0;
    res(5).vote=0;
    res(6).vote=0;
    res(7).vote=0;
    res(8).vote=0;
    res(9).vote=0;
    res(10).vote=0;
    temmp = zeros(10,1);
    for j=1:(cc-1)
        pred = svmPredict(model(j).model,test_samples(i,:));
        if pred == 0
            res(model(j).i).vote = res(model(j).i).vote + 1;
        else
            res(model(j).j).vote = res(model(j).j).vote + 1;
        end
    end
    for k = 1:10
        temmp(k) = res(k).vote;
    end
    bigm = max(temmp);
    idx = find(temmp==bigm,1);
    if idx ==10
        idx=0;
    end
    result(numbercount) = idx;
    numbercount = numbercount +1;
end
correctcount = 0; 
for i=1:numbercount-1
    if result(i)==test_samples_labels(i)
        correctcount = correctcount + 1;
    end
end
fprintf("Accuracy is:")
disp(correctcount/Ntest)
end

