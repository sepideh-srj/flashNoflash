function [crit,x] = getBrightness(image,wp)
    
    [width,height,~] = size(image);
    num = width*height/3;
    image = xyz2lab(image,'WhitePoint', wp);
    lumimage=image(:,:,1);
%     
%     figure;
%     imshow(lumimage,[])
%     figure;
%     histogram(lumimage,10)
% %     saveas(gcf,name)
%     maxB = max(max(lumimage(:)));   
    B = maxk(lumimage(:),2000);
    minB = min(B);
    x = ceil(61 / minB);
    
%     [~,idx2] = mink(lumimage(:),num);
%     [row2,col2] = ind2sub(size(lumimage),idx2);
%     [row,col] = ind2sub(size(lumimage),idx);
    
%     X1=0;
%     
%     X2=0;
%     
%     for i=1:numel(row)
%         X1 = X1+lumimage(row(i),col(i));
% 
%         X2 = X2+lumimage(row2(i),col2(i));
% 
%     end
% %     [L,Centers] = imsegkmeans(lumimage,3);
% %     B = labeloverlay(lumimage,L);
% %     figure;
% %     imshow(B)
% %     title('Labeled Image')
    crit = sum(sum( lumimage > 60));
%     luminance1 = X1/num;
%     luminance2 = X2/num;
%     stIm = std(std(lumimage))
end