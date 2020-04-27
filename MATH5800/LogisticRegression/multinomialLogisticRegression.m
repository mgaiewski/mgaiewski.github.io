function [Time,Accuracy,TestLabels,TestValues,TestGuesses,Error,Theta] = multinomialLogisticRegression(Percentage,MaxGradientIterations,Tolerance)
%INPUTS:
%Percentage = percentage of the data to train the model to
%MaxGradientIterations = maximum number of iterations the gradient search
%%%will take to find regression coefficients
%Tolerance = "Epsilon" for Cauchy convergence

%OUTPUTS:
%Time = How long the algorithm ran for in seconds
%Accuracy = percentage of the test data that the regression correctly found
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

LearningRate = 0.001;

%Perform one iteration of Gradient Search
%Speed this way up
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
    
    %Speed this way up
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
    MaxDifference = max(MaxDifference)
    Count = Count + 1
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
TestValues = TestNumerators./TestDenominators;

%Calculate Test Values
for i = 1:TestSize
    [~,Element] = max(TestValues(i,:));
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