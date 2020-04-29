%Johannes Langelaar (2020). MNIST neural network training and testing
%(https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural
%-network-training-and-testing), MATLAB Central File Exchange.
%Retrieved March 11, 2020.
%Edited

function [Weight12,Weight23,Weight34,Bias12,Bias23,Bias34] = digitTrain(Percentage,LearningRate,HiddenNeurons1,HiddenNeurons2,Epochs,MinibatchSize)

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
TestPixels = Data(TrainSize + 1:DataSize,2:785)./255;

Output = zeros(10,TrainSize); %Correct outputs vector
for i = 1:TrainSize
    Output(TrainLabels(i)+1,i) = 1;
end

%Initializing weights and biases
Weight12 = randn(HiddenNeurons1,784)*sqrt(2/784);
Weight23 = randn(HiddenNeurons2,HiddenNeurons1)*sqrt(2/HiddenNeurons1);
Weight34 = randn(10,HiddenNeurons2)*sqrt(2/HiddenNeurons2);
Bias12 = randn(HiddenNeurons1,1);
Bias23 = randn(HiddenNeurons2,1);
Bias34 = randn(10,1);

for k = 1:Epochs %Outer epoch loop
    
    Batches = 1;
    
    for j = 1:TrainSize/MinibatchSize
        ErrorTo4 = zeros(10,1);
        ErrorTo3 = zeros(HiddenNeurons2,1);
        ErrorTo2 = zeros(HiddenNeurons1,1);
        Grad4 = zeros(10,1);
        Grad3 = zeros(HiddenNeurons2,1);
        Grad2 = zeros(HiddenNeurons1,1);
        for i = Batches:Batches+MinibatchSize-1 %Loop over each minibatch
            
            %Feed forward
            A1 = TrainPixels(:,i);
            Z2 = Weight12*A1 + Bias12;
            A2 = elu(Z2);
            Z3 = Weight23*A2 + Bias23;
            A3 = elu(Z3);
            Z4 = Weight34*A3 + Bias34;
            A4 = elu(Z4); %Output vector
            
            %backpropagation
            Error4 = (A4-Output(:,i)).*elup(Z4);
            Error3 = (Weight34'*Error4).*elup(z3);
            Error2 = (Weight23'*Error3).*elup(z2);
            
            ErrorTo4 = ErrorTo4 + Error4;
            ErrorTo3 = ErrorTo3 + Error3;
            ErrorTo2 = ErrorTo2 + Error2;
            Grad4 = Grad4 + Error4*A3';
            Grad3 = Grad3 + Error3*a2';
            Grad2 = Grad2 + Error2*A1';
        end
        
        %Gradient descent
        Weight34 = Weight34 - LearningRate/MinibatchSize*Grad4;
        Weight23 = Weight23 - LearningRate/MinibatchSize*Grad3;
        Weight12 = Weight12 - LearningRate/MinibatchSize*Grad2;
        Bias34 = Bias34 - LearningRate/MinibatchSize*ErrorTo4;
        Bias23 = Bias23 - LearningRate/MinibatchSize*ErrorTo3;
        Bias12 = Bias12 - LearningRate/MinibatchSize*ErrorTo2;
        
        Batches = Batches + MinibatchSize;
        
    end
    [TrainPixels,Output] = shuffle(TrainPixels,Output); %Shuffles order of the images for next epoch
end