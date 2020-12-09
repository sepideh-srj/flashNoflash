% datadir     = 'Illuminations_Samples';    %the directory containing the images
% resultsdir  = 'resized'; %the directory for dumping results
% imglist = dir(sprintf('%s/*.png', datadir));
% numel(imglist)
images1 ='flash';
images2 ='ambient';
pngFiles1=dir(fullfile(images1,'/*.png*'));
pngFiles2=dir(fullfile(images2,'/*.png*'));
destinationFolder = 'Aval';
if ~exist(destinationFolder, 'dir')
  mkdir(destinationFolder);
end
destinationFolder2 = 'Bval';
if ~exist(destinationFolder2, 'dir')
  mkdir(destinationFolder2);
end
destinationFolder3 = 'Atest';
if ~exist(destinationFolder3, 'dir')
  mkdir(destinationFolder3);
end
destinationFolder4 = 'Btest';
if ~exist(destinationFolder4, 'dir')
  mkdir(destinationFolder4);
end

for i=1:150
im=pngFiles1((i-1)*10+1).name;
im2=pngFiles2((i-1)*10+1).name;
image=imread(fullfile(images1,im));
image2=imread(fullfile(images2,im));
fullDestinationFileName2 = fullfile(destinationFolder2, im);
fullDestinationFileName1 = fullfile(destinationFolder, im);
fullFileName1 = fullfile(images1, im);
fullFileName2 = fullfile(images2, im);

movefile(fullFileName1, destinationFolder)
movefile(fullFileName2, destinationFolder2)
end


for i=1:150
im=pngFiles1((i-1)*10+2).name
im2=pngFiles2((i-1)*10+2).name
image=imread(fullfile(images1,im));
image2=imread(fullfile(images2,im));
fullDestinationFileName2 = fullfile(destinationFolder2, im);
fullDestinationFileName1 = fullfile(destinationFolder, im);
fullFileName1 = fullfile(images1, im);
fullFileName2 = fullfile(images2, im);

movefile(fullFileName1, destinationFolder3)
movefile(fullFileName2, destinationFolder4)
end


