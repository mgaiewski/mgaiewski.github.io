%Johannes Langelaar (2020). MNIST neural network training and testing 
%(https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural
%-network-training-and-testing), MATLAB Central File Exchange. 
%Retrieved March 11, 2020.

function [B,v] = shuffle(A,y)
cols = size(A,2);
P = randperm(cols);
B = A(:,P);
v = y(:,P);