%Johannes Langelaar (2020). MNIST neural network training and testing
%(https://www.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural
%-network-training-and-testing), MATLAB Central File Exchange. Retrieved
%March 11, 2020.

function [f] = elup(x)
f = zeros(length(x),1);
for i = 1:length(x)
    if x(i)>=0
        f(i) = 1;
    else
        f(i) = 0.2*exp(x(i));
    end
end