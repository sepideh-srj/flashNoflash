% datadir     = 'Illuminations_Samples';    %the directory containing the images
% resultsdir  = 'resized'; %the directory for dumping results
% imglist = dir(sprintf('%s/*.png', datadir));
% numel(imglist)
close all;
images ='All';
% test = imread('temp.png');
pngFiles=dir(fullfile(images,'/*.png*'));
destinationFolder = 'good';
if ~exist(destinationFolder, 'dir')
  mkdir(destinationFolder);
end
destinationFolder2 = 'bad';
if ~exist(destinationFolder2, 'dir')
  mkdir(destinationFolder2);
end
numel(pngFiles)
dataGood = [];
dataBad = [];
for i=1 :numel(pngFiles)/2
    im=pngFiles(2*i).name;
    im2 = pngFiles(2*i-1).name;
    flash=imread(fullfile(images,im));
    
    ambient=imread(fullfile(images,im2));
    rawFile = imfinfo(fullfile(images,im));
    flash = imresize(flash,0.3);
    ambient = imresize(ambient,0.3);
     name = sprintf('%d.png', i);

    matrix =str2num(rawFile.Comment); 
    des = str2num(rawFile.Description);
    flash2 = getXYZ(flash, matrix);
    ambient2 = getXYZ(ambient, matrix);
    [minF, maxF, stIm, crit] = getBrightness(flash2, illuminantCode(des), im);
    
    if (crit > 1500)
        dataGood(end+1,:) = [minF, maxF, i, crit];
        fullDestinationFileName2 = fullfile(destinationFolder, im);
        imwrite (flash,fullDestinationFileName2,'Comment',rawFile.Comment,'Description', rawFile.Description)
        fullDestinationFileName = fullfile(destinationFolder, im2);        
        imwrite (ambient,fullDestinationFileName,'Comment',rawFile.Comment,'Description', rawFile.Description)
    else
        dataBad(end+1,:) = [minF, maxF, i, crit];
        fullDestinationFileName2 = fullfile(destinationFolder2, im);
        imwrite (flash,fullDestinationFileName2,'Comment',rawFile.Comment,'Description', rawFile.Description)
        fullDestinationFileName = fullfile(destinationFolder2, im2);        
        imwrite (ambient,fullDestinationFileName,'Comment',rawFile.Comment,'Description', rawFile.Description)
        
    end

% %     if (des == 17)
% %         flash2 = changeTemperature2(flash);
% %     end
% 
%      flash = xyztorgb(ambient,des);

%      flash = flash(1:323, 1:430,:);
%      test =  test(1:324, 1:432,:);
%      K = imabsdiff(flash,test);
%      imshow(K(:,:,1))
%     ambient = changeBrightness(ambient, illuminantCode(des));
%     flash2 = xyztorgb(flash,des);

%     flash2 = changeTemperature2(flash);
%     flash3 = xyztorgb(flash2,des);
% 
%     figure()
%     imshow(flash3)
%     figure()
%     imshow(flash)
%     output = (flash2*0.8+ ambient*1.4);
%       input = flash + ambient;
%       output = flash * 2;
% %     const = 80;
% %     randNum = randi([0 200])  
% %     input = flash*randNum/100 + ambient*(220-randNum)/100;
% %     if randNum <= 200 - const
% %         randNum2 = const + randNum;
% %     else
% %         randNum2 = 200;
% %     end
% %      output = flash*randNum2/100 + ambient*(220-randNum2)/100;
%     input = xyz2rgb(input, 'WhitePoint', illuminantCode(des));
%     output = xyz2rgb(output, 'WhitePoint', illuminantCode(des));
%     output = real(output);
%     
%     input = xyztorgb(input,des);
%     input = real(input);
%     ratio = output ./ input
%     imshow([input,output, ratio])

end


