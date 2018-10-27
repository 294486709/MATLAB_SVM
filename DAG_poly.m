function DAG_poly()
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
% Try different SVM Parameters here
%[C, sigma] = dataset3Params(train_samples, train_samples_labels, test_samples, test_samples_labels);
model = struct('i',{},'j',{},'model',{});
res = struct('vote',{});
C = 1;
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
        model(cc).model = svmTrain(training, training_label, C, @(x1, x2) PolynominalKernel(x1, x2));
        cc = cc + 1;
        disp(cc-1)
    end
end

%predictions = svmPredict(model_0, test_samples);
Ntest = size(test_samples,1);
numbercount=1;
ilist = zeros(45,1);
jlist = zeros(45,1);
for i =1:45
    if model(i).i==10
        model(i).i = 0;
    end
    ilist(i) = model(i).i;
    jlist(i) = model(i).j;
end


result = zeros(Ntest,1);
for i=1:Ntest
    currentsample = test_samples(i);
    leftbound = 0;
    rightbound = 9;
    while 1
        invf=0;
        leftidx = find(ilist==leftbound);
        rightidx = find(jlist==rightbound);
        realidx = intersect(leftidx,rightidx);
        if isempty(realidx)
            leftidx = find(jlist == leftbound);
            rightidx = find(ilist == rightbound);
            realidx = intersect(leftidx,rightidx);
        end
        pred = svmPredict(model(realidx).model,test_samples(i,:));
        leftbound = model(realidx).i;
        rightbound = model(realidx).j;
        if leftbound > rightbound
            t = leftbound;
            leftbound = rightbound;
            rightbound = t;
            invf=1;
        end
        if invf ==0
            if pred == 0
                rightbound = rightbound - 1;
            else
                leftbound = leftbound + 1;
            end
        else
            if pred == 0
                leftbound = leftbound + 1;
            else
                rightbound = rightbound - 1;
            end
            
        end
        if leftbound == rightbound
            result(i) = leftbound;
            break
        end
        
        
    end
end
correctcount = 0;
for i=1:1000
    if result(i)==test_samples_labels(i)
        correctcount = correctcount + 1;
    end
end
fprintf("Accuracy is:")
disp(correctcount/Ntest)



end