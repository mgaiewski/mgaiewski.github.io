function trackThetaPictures(Theta)

%Visualize the digits 0-9 in the logistic regression
Pixels = zeros(28,280);
Theta = reshape(Theta,[7840,1]);
for j = 1:10 %digits
    Shift = 28*(j-1);
    for i = 1:28%rows
        for k =1:28 %little columns
            Pixels(i,k + Shift) = Theta(784*(j-1) + 28*(i-1) + k);
        end
    end
end
rescale(Pixels);
Pixels = 255.*Pixels;
InvertColors = 255*ones(28,280);
Image = mat2gray(InvertColors-round(Pixels),[0,255]);
imshow(Image,'InitialMagnification',500);