function [Network,Time,Accuracy,TestGuesses,TestLabels,TestProbabilities] = convolutionalNeuralNetwork(Portion,MaximumEpochs,LearningRate)
%INPUTS:
%Portion = how much of the data to train (between 0 and 1)
%Maximum Epochs = maximum number of Epochs
%LearningRate = learning rate for gradient descent

%OUTPUTS:
%Network = Convolutional Neural Network
%Time = time it took the algorithm to run
%Accuracy = how accurate the model was on the test data 
%TestGuesses = what the model predicted each image was
%TestLabels = what the correct value of the image was
%TestProbabilities = the probabilities of each digit 0-9
%%%%%

%This is MATLAB's own example with MNIST
%https://www.mathworks.com/help/deeplearning/ug/create-simple-deep
%-learning-network-for-classification.html
%Edited by Michael Gaiewski

tic
%Download MNIST Data
DigitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
Data = imageDatastore(DigitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

TrainSize = round(0.1*Portion*size(Data.Files,1));
[TrainData,TestData] = splitEachLabel(Data,TrainSize,'randomize');

%Specify CNN Layers
Layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

Options = trainingOptions('sgdm', ...
    'InitialLearnRate',LearningRate, ...
    'MaxEpochs',MaximumEpochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',TestData, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train Network
Network = trainNetwork(TrainData,Layers,Options);

%Test Data
[TestGuesses,TestProbabilities] = classify(Network,TestData);
TestLabels = TestData.Labels;

%Calculate Accuracy
Accuracy = sum(TestGuesses == TestLabels)/numel(TestLabels);
Time = toc;