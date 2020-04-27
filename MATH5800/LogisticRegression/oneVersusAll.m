function [Time,Accuracy] = oneVersusAll(Percentage,MaxGradientIterations,Tolerance)
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

%Preallocation
TestMatrix = zeros(TestSize,9);
TestValues = zeros(TestSize,1);
Theta = zeros(784,10);
TempData = zeros(TrainSize,785);

%Split Data into train set and test set
TrainData = Data(1:TrainSize,:);

%Distinguish between data and labels
TrainLabels = TrainData(1:TrainSize,1);
TestLabels = Data(TrainSize +1:DataSize,1);
TestDigits = Data(TrainSize+1:DataSize,2:785);

%Calculate number of each digit in training set
NumberOfs = zeros(10,1);
for i = 1:10
    NumberOfs(i) = sum(TrainLabels(:) == i-1);
end

for i = 1:10
    %Distinguish between digit and everything else in data
    YesDigits = zeros(NumberOfs(i),785);
    NoDigits = zeros(TrainSize- NumberOfs(i),785);
    YesRow = 1;
    NoRow=1;
    for j = 1:TrainSize
        if(TrainLabels(j) == i-1)
            YesDigits(YesRow,:) = TrainData(j,:);
            YesRow = YesRow + 1;
        else
            NoDigits(NoRow,:) = TrainData(j,:);
            NoRow = NoRow + 1;
        end
    end
    YesDigits(:,1) = 1;
    NoDigits(:,1) = 0;
    TempData(1:NumberOfs(i),:) = YesDigits;
    TempData(NumberOfs(i)+ 1:TrainSize,:) = NoDigits;
    TempLabels = TempData(:,1);
    TempDigits = TempData(:,2:785);
    
    %Perform one iteration Gradient Search for Logistic Regression Coefficients
    TempTheta = zeros(784,1);
    Train = 1./(1+exp(-TempDigits*TempTheta));
    Grad = (1/784)*(TempDigits'*(Train-TempLabels));
    NewTheta = TempTheta - 0.01*Grad;
    MaxDifference = max(abs(NewTheta));
    Count = 1;
    
    %Continue iterations until Cauchy convergence or maximum iterations are met
    while(MaxDifference > Tolerance && Count <= MaxGradientIterations)
        OldTheta = NewTheta;
        Train = 1./(1+exp(-TempDigits*OldTheta));
        Grad = (1/784)*(TempDigits'*(Train-TempLabels));
        NewTheta = OldTheta - 0.01*Grad;
        MaxDifference = max(abs(0.01*Grad));
        Count = Count + 1;
    end
    Theta(:,i) = NewTheta;
end

%Calculate Test Values
for i = 1:TestSize
    Input = TestDigits(i,:);
    for j = 1:10
        TestMatrix(i,j) = 1/(1+ exp(-dot(Input,Theta(:,j))));
    end
    [~,Element] = max(TestMatrix(i,:));
    TestValues(i) = (Element-1);
end

%Determine Accuracy
Error = abs(TestLabels - TestValues);
for i = 1:TestSize
    if(Error(i) > 1)
        Error(i) = 1;
    else
        Error(i) = 0;
    end
end
Accuracy = 1 - (sum(Error)/size(Error,1));

Time = toc;