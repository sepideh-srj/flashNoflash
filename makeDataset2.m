% datadir     = 'Illuminations_Samples';    %the directory containing the images
% resultsdir  = 'resized'; %the directory for dumping results
% imglist = dir(sprintf('%s/*.png', datadir));
% numel(imglist)
images ='good';

pngFiles=dir(fullfile(images,'/*.png*'));
destinationFolder = 'flash';
if ~exist(destinationFolder, 'dir')
  mkdir(destinationFolder);
end
destinationFolder2 = 'ambient';
if ~exist(destinationFolder2, 'dir')
  mkdir(destinationFolder2);
end


for i=1:(numel(pngFiles))/2
    im=pngFiles(2*i).name;
    im2 = pngFiles(2*i-1).name;
    flash=imread(fullfile(images,im));
    ambient=imread(fullfile(images,im2));
    rawFile = imfinfo(fullfile(images,im));
%     flash = imresize(im1,0.3);
%     ambient = imresize(im2,0.3);
%     name = sprintf('%d.png', i);
    matrix =str2num(rawFile.Comment); 
    des = str2num(rawFile.Description);
    flash = getXYZ(flash, matrix);
    
    ambient = getXYZ(ambient, matrix);
    ambient = changeBrightness(ambient, illuminantCode(des));
%     flash2 = changeTemperature2(flash);
%      flash2 = xyz2rgb(flash2, 'WhitePoint', illuminantCode(des));
%     figure()
%     imshow(flash2)
%     
%     flash = xyz2rgb(flash, 'WhitePoint', illuminantCode(des));
    
%     ambient = changeBrightness(ambient, illuminantCode(des));
%     ambient = changeTemperature(ambient, illuminantCode(des));
%     output = flash + ambient;
%     output = xyz2rgb(output, 'WhitePoint', illuminantCode(des));
    name = sprintf('%d.png', i);
    fullDestinationFileName2 = fullfile(destinationFolder2, name);
%     input = ambient;
%     input = xyz2rgb(input, 'WhitePoint', illuminantCode(des));
    imwrite (ambient,fullDestinationFileName2,'Comment',rawFile.Comment,'Description', rawFile.Description);
    fullDestinationFileName = fullfile(destinationFolder, name);
%     [width,height,~] = size(combImg)
%     nameFile = sprintf('%d.mat', 2*i-1);
    imwrite (flash,fullDestinationFileName,'Comment',rawFile.Comment,'Description', rawFile.Description);
    
end


