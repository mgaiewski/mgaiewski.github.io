%PictureTest1

%Read MNIST Data
ExcelMNIST = csvread('train.csv',1,0);
Labels = ExcelMNIST(:,1);
Digits = ExcelMNIST(:,2:785);%./255;


%1. Train the conv nn
[Network,Time,Accuracy,TestGuesses,TestLabels,TestProbabilities] = convolutionalNeuralNetwork(0.5,10,0.01);

%2. "get hold of" the 8 3x3 matrices in net.Layers(2).Weights
Layer2Weights = Network.Layers(2,1).Weights;

%3. Write a function using convn(A,B,'same') that takes an image from the 
%training set (call that A, it's a 28x 28 matrix) and computes convn(A,f) 
%for each of the 8 filters f that you in Step 2.
A = Digits(764,:);
A = reshape(A,[28 28]);
C = convn(A,Layer2Weights);
C = rescale(C);


%4. Now you have 8 28x28 arrays that have positive and negative entries -- 
%you want to display these as images, so you either can drop all the 
%negative entrires and rescale to 0->255 or just rescale from 0->255
%or somehow make it an image.

Pixels = zeros(28,224);
InvertColors = ones(28,224);
for i = 1:8
    Shift = 28*(i-1);
    Pixels(:,Shift+1:Shift+28) = C(2:29,2:29,1,i);%ask about 2:29
end
    

Image = mat2gray(InvertColors-Pixels); 
imshow(Pixels,'InitialMagnification',500);

%5. 1 row for each image; the first image in the row is the original image,
%the next 8 images are the results of the cnvolution with each of the 8 
%filters.
