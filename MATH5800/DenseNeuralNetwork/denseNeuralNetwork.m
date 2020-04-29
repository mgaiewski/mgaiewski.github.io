function [Time,Accuracy,TestLabels,TestValues,TestGuesses,Error,Theta] = denseNeuralNetwork(Percentage,LearningRate,HiddenNeurons1,HiddenNeurons2,Tolerance,MaxEpochs,BatchSize)

%Johannes Langelaar (2020). MNIST neural network training and testing
%(https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural
%-network-training-and-testing), MATLAB Central File Exchange.
%Retrieved March 11, 2020.
%Edited by Michael Gaiewski

%INPUTS:
%Percentage = percentage of data to use for training(between 0 and 1)
%LearningRate = learning rate for gradient descent
%HiddenNuerons1 = Number of hidden neurons in the first hidden layer
%HiddenNuerons2 = Number of hidden neurons in the second hidden layer
%Tolerance = Cauchy Convergence criteria for gradient descent (positive)
%MaxEpochs = Maximum Number of Epochs
%BatchSize = Number of entries to use for each batch

%OUTPUTS:
%Time = time it took the algorithm to run
%Accuracy = how accurate the model was on the test data 
%TestGuesses = what the model predicted each image was
%TestLabels = what the correct value of the image was
%Error = Vector showing which of the data points are right and wrong
%Theta = Weight matrix multiplied for picture (not exactly useful)
%%%%%

tic
Data = csvread('train.csv',1,0);
DataSize = size(Data,1);
TrainSize = round(Percentage*DataSize);
TestSize = DataSize - TrainSize;
Data = Data(randperm(DataSize),:);

%Split Data into train set and test set
TrainData = Data(1:TrainSize,:);

%Distinguish between data and labels
TrainLabels = TrainData(1:TrainSize,1);
TrainPixels = (TrainData(1:TrainSize,2:785)./255)';
TestLabels = Data(TrainSize + 1:DataSize,1);
TestPixels = (Data(TrainSize + 1:DataSize,2:785)./255)';
TestValues = zeros(TestSize,10);
TestGuesses = zeros(TestSize,1);

Output = zeros(10,TrainSize); %Correct outputs vector
for i = 1:TrainSize
    Output(TrainLabels(i)+1,i) = 1;
end

%Initializing weights and biases
Weight12 = zeros(HiddenNeurons1,784);
Weight23 = zeros(HiddenNeurons2,HiddenNeurons1);
Weight34 = zeros(10,HiddenNeurons2);
Bias12 = randn(HiddenNeurons1,1);
Bias23 = randn(HiddenNeurons2,1);
Bias34 = randn(10,1);

MaxDifference = Inf;
Epoch = 1;

while(MaxDifference > Tolerance && Epoch < MaxEpochs)
    Batches = 1;
    for j = 1:TrainSize/BatchSize
        ErrorTo4 = zeros(10,1);
        ErrorTo3 = zeros(HiddenNeurons2,1);
        ErrorTo2 = zeros(HiddenNeurons1,1);
        Grad4 = zeros(10,1);
        Grad3 = zeros(HiddenNeurons2,1);
        Grad2 = zeros(HiddenNeurons1,1);
        for k = Batches:Batches+BatchSize-1
            
            %Feed forward
            A1 = TrainPixels(:,k);
            Z2 = Weight12*A1 + Bias12;
            A2 = elu(Z2);
            Z3 = Weight23*A2 + Bias23;
            A3 = elu(Z3);
            Z4 = Weight34*A3 + Bias34;
            A4 = elu(Z4); %Output vector
            
            %backpropagation
            Error4 = (A4-Output(:,k)).*elup(Z4);
            Error3 = (Weight34'*Error4).*elup(Z3);
            Error2 = (Weight23'*Error3).*elup(Z2);
            
            ErrorTo4 = ErrorTo4 + Error4;
            ErrorTo3 = ErrorTo3 + Error3;
            ErrorTo2 = ErrorTo2 + Error2;
            Grad4 = Grad4 + Error4*A3';
            Grad3 = Grad3 + Error3*A2';
            Grad2 = Grad2 + Error2*A1';
        end
        
        %Gradient descent
        Weight34 = Weight34 - LearningRate/BatchSize*Grad4;
        Weight23 = Weight23 - LearningRate/BatchSize*Grad3;
        Weight12 = Weight12 - LearningRate/BatchSize*Grad2;
        Bias34 = Bias34 - LearningRate/BatchSize*ErrorTo4;
        Bias23 = Bias23 - LearningRate/BatchSize*ErrorTo3;
        Bias12 = Bias12 - LearningRate/BatchSize*ErrorTo2;
        
        Batches = Batches + BatchSize;
    end
    MaxDifference = max(abs(LearningRate.*Grad4));
    MaxDifference = max(MaxDifference);
    Epoch = Epoch + 1;
    Theta = (Weight12')*(Weight23')*(Weight34');
    trackThetaPictures(Theta);
    [TrainPixels,Output] = shuffle(TrainPixels,Output); %Shuffles order of the images for next epoch
end

%Test Data
for i = 1:TestSize
     Output2 = elu(Weight12*TestPixels(:,i)+Bias12);
     Output3 = elu(Weight23*Output2+Bias23);
     TestValues(i,:) = elu(Weight34*Output3+Bias34)';
  
    Biggest = -Inf;
    for k = 1:10
        if (TestValues(i,k) > Biggest)
            TestGuesses(i) = k-1;
            Biggest = TestValues(i,k);
        end
    end
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