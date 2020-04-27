function [Network,Time,Accuracy,TestGuesses,TestLabels,TestProbabilities] = convolutionalNeuralNetwork(Portion,MaximumEpochs,LearningRate)
%This is MATLAB's own example with MNIST

tic

DigitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
Data = imageDatastore(DigitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

TrainSize = round(0.1*Portion*size(Data.Files,1));
[TrainData,TestData] = splitEachLabel(Data,TrainSize,'randomize');

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

Network = trainNetwork(TrainData,Layers,Options);
[TestGuesses,TestProbabilities] = classify(Network,TestData);
TestLabels = TestData.Labels;

Accuracy = sum(TestGuesses == TestLabels)/numel(TestLabels);
Time = toc;