function [Results] = binaryLogisticRegression(Percentage,MaxGradientIterations,Tolerance)
%INPUTS:
%Percentage = percentage of the data to train the model to
%MaxGradientIterations = maximum number of iterations the gradient search
%%%will take to find regression coefficients
%Tolerance = "Epsilon" for Cauchy convergence

%OUTPUTS:
%Results = Accuracy, time, and confusion matrix
%%%%%

tic

%Read Binary 0-1 Data, get sizes of matrices and randomize rows to not
%%%depend on order
Data = csvread('train01.csv',0,0);
DataSize = size(Data,1);
TrainSize = round(Percentage*DataSize);
TestSize = DataSize - TrainSize;
Data = Data(randperm(DataSize),:);

%Preallocation
TestValues = zeros(TestSize,1);
Theta = zeros(784,1);

%Split Data into train set and test set
%Distinguish between data and labels
TrainLabels = Data(1:TrainSize,1);
TrainDigits = Data(1:TrainSize,2:785);
TestLabels = Data(TrainSize +1:DataSize,1);
TestDigits = Data(TrainSize+1:DataSize,2:785);

%Perform one iteration Gradient Search for Logistic Regression Coefficients
Train = 1./(1+exp(-TrainDigits*Theta));
Grad = (TrainDigits'*(Train-TrainLabels));
NewTheta = Theta - 0.01*Grad;
MaxDifference = max(abs(NewTheta));
Count = 1;

%Continue iterations until Cauchy convergence or maximum iterations are met
while(MaxDifference > Tolerance && Count <= MaxGradientIterations)
    OldTheta = NewTheta;
    Train = 1./(1+exp(-TrainDigits*OldTheta));
    Grad = (TrainDigits'*(Train-TrainLabels));
    NewTheta = OldTheta - 0.01*Grad;
    MaxDifference = max(abs(0.01*Grad));
    Count = Count + 1;
end
Theta = NewTheta;

%Calculate Test Values
for i = 1:TestSize
    Input = TestDigits(i,:);
    TestValues(i) = 1/(1+ exp(-dot(Input,Theta)));
end

Time = toc;

%Determine Accuracy
TestValues = round(TestValues);
Error = abs(TestLabels - TestValues);
Accuracy = 1 - (sum(Error)/size(Error,1));

%Make Confusion Matrix
Results = cell(6,3);
Correct0 = 0;
Correct1 = 0;
Incorrect0= 0;
Incorrect1= 0;
for i = 1:TestSize
   if(TestLabels(i) == 0 && TestValues(i) == 0)
       Correct0 = Correct0+1;
   elseif(TestLabels(i) == 1 && TestValues(i) == 1)
       Correct1 = Correct1+1;
   elseif(TestLabels(i) == 0 && TestValues(i) == 1)
       Incorrect1 = Incorrect1+1;
   else
       Incorrect0 = Incorrect0+1;
   end
end
Results{2,2} = Correct0;
Results{3,2} = Incorrect0;
Results{2,3} = Incorrect1;
Results{3,3} = Correct1;

%Label Results
Results{1,1} = 'Confusion Matrix';
Results{2,1} = 'True = 0';
Results{3,1} = 'True = 1';
Results{1,2} = 'Predicted = 0';
Results{1,3} = 'Predicted = 1';
Results{5,1} = ['Accuracy = ',num2str(100*Accuracy),'%'];
Results{6,1} = ['Time = ',num2str(Time),' seconds'];