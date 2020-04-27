function [Time,Accuracy,TestLabels,TestProbabilities,TestGuesses,Error,Theta] = multinomialLogisticRegression(Percentage,MaxGradientIterations,LearningRate,Tolerance)
%INPUTS:
%Percentage = percentage of data to use for training(between 0 and 1)
%MaxGradientIterations = Maximum number of iterations for gradient descent
%LearningRate = learning rate for gradient descent
%Tolerance = Cauchy Convergence criteria for gradient descent (positive)

%OUTPUTS:
%Time = time it took the algorithm to run
%Accuracy = how accurate the model was on the test data 
%TestLabels = what the correct value of the image was
%TestProbabilities = probabilities of each digit on the test data
%TestGuesses = what the model predicted each image was
%Error = Vector showing which of the data points are right and wrong
%Theta = Regression coefficient matrix for picture
%%%%%

tic

%Read Train Data, get sizes of matrices and randomize rows to not depend on
%%%order
Data = csvread('train.csv',1,0);
DataSize = size(Data,1);
TrainSize = round(Percentage*DataSize);
TestSize = DataSize - TrainSize;
Data = Data(randperm(DataSize),:);

%Split Data into train set and test set
TrainData = Data(1:TrainSize,:);

%Distinguish between data and labels
TrainLabels = TrainData(1:TrainSize,1);
TrainDigits = TrainData(1:TrainSize,2:785)./255;
TestLabels = Data(TrainSize + 1:DataSize,1);
TestDigits = Data(TrainSize + 1:DataSize,2:785)./255;

%Preallocation
TrainNumerators = zeros(TrainSize,10);
TrainDenominators = zeros(TrainSize,10);
Theta = zeros(784,10);
Grad = zeros(784,10);
TestGuesses = zeros(TestSize,1);
TestNumerators = zeros(TestSize,10);
TestDenominators = zeros(TestSize,10);

%Perform one iteration of Gradient Search
for j = 1:10
    for i = 1:TrainSize
        TrainNumerators(i,j) = exp(TrainDigits(i,:)*Theta(:,j));
    end
end
TrainDenominatorSum = sum(TrainNumerators,2);
for i = 1:TrainSize
    TrainDenominators(i,:) = TrainDenominatorSum(i);
end
TrainProbabilities = TrainNumerators./TrainDenominators;

for j = 1:10
    for i = 1:TrainSize
        if(TrainLabels(i) == j-1)
            Grad(:,j) = Grad(:,j) + TrainDigits(i,:)';
        end
        Grad(:,j) = Grad(:,j) - TrainDigits(i,:)'.*TrainProbabilities(i,j);
    end
end
%Grad = Grad./TrainSize;
NewTheta = Theta + LearningRate.*Grad;
MaxDifference = max(abs(NewTheta));
MaxDifference = max(MaxDifference);
Count = 1;

%Continue iterations until Cauchy convergence or maximum iterations are met
while(MaxDifference > Tolerance && Count < MaxGradientIterations)
    Grad = zeros(784,10);
    OldTheta = NewTheta;
    for j = 1:10
        for i = 1:TrainSize
            TrainNumerators(i,j) = exp(TrainDigits(i,:)*OldTheta(:,j));
        end
    end
    TrainDenominatorSum = sum(TrainNumerators,2);
    for i = 1:TrainSize
        TrainDenominators(i,:) = TrainDenominatorSum(i);
    end
    TrainProbabilities = TrainNumerators./TrainDenominators;
    
    for j = 1:10
        for i = 1:TrainSize
            if(TrainLabels(i) == j-1)
                Grad(:,j) = Grad(:,j) + TrainDigits(i,:)';
            end
            Grad(:,j) = Grad(:,j) - TrainDigits(i,:)'.*TrainProbabilities(i,j);
        end
    end
    MaxDifference = max(abs(LearningRate.*Grad));
    MaxDifference = max(MaxDifference);
    Count = Count + 1;
    NewTheta = OldTheta + LearningRate.*Grad;
    trackThetaPictures(NewTheta);
end
Theta = NewTheta;

%Calculate Test Values
for j = 1:10
    for i = 1:TestSize
        TestNumerators(i,j) = exp(TestDigits(i,:)*Theta(:,j));
    end
end

TestDenominatorSum = sum(TestNumerators,2);
for i = 1:TestSize
    TestDenominators(i,:) = TestDenominatorSum(i);
end
TestProbabilities = TestNumerators./TestDenominators;

%Calculate Test Values
for i = 1:TestSize
    [~,Element] = max(TestProbabilities(i,:));
    TestGuesses(i) = Element-1;
end

%Determine Accuracy
Error = abs(TestLabels - TestGuesses);
for i = 1:TestSize
    if(Error(i) >= 1)
        Error(i) = 1;
    else
        Error(i) = 0;
    end
end
Accuracy = 1 - (sum(Error)/size(Error,1));

Time = toc;