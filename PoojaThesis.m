% 1- orthogonal planes to separate jaw
% 2- superimpose
% 3- length of jaw (sliceomatic?)
% 4- volume using points
% try to fix segmentation/ image processing
% try to pick points by editing code for volshow

% THIS IS OLD CODE
%% Creating 3d Array and bone data
%inputs: Bone_cutoff, Image to read name/path

% Input variables
Bone_cutoff = 1200;
I_name = 'VOL_KEERTHI_BEFORE';
I_post_name = 'VOL_KEERTHI_AFTER';

% Reading image
% I = dicomread('3DSlice1.dcm');
% imshow(I,'DisplayRange',[])
V = dicomreadVolume(I_name);
V = squeeze(V);
Vsingle = im2single(V);
%figure; volshow(V)
V_post = dicomreadVolume(I_post_name);
V_post = squeeze(V_post);
%figure; volshow(V_post)

% Creating binary image of bone
V_all_bone = single(V>Bone_cutoff);
V_post_bone = zeros(size(V_all_bone));
dim_post = size(V_post);
V_post_bone(1:dim_post(1),1:dim_post(2),2:(dim_post(3)+1))=single(V_post>Bone_cutoff);

V_intersection = 1*single(V_all_bone & V_post_bone);
V_pre_only = 5*single(xor(V_all_bone, V_post_bone) & V_all_bone);
V_post_only = 250*single(xor(V_all_bone, V_post_bone) & V_post_bone);
V_together = V_intersection + V_pre_only + V_post_only;

diffdisplay(V_together);

%% Superimpose images % CHECK INITIAL TRANSFORMATION,OPTIONAL!!!!!
                        
% how to store header information? need struct with 
% SliceThickness: _  --- in DICOM header, 
% PixelSize: [_,_]   --- in DICOM header, PixelSpacings (0.3,0.3)

% Translation:
% [1  0   0   0]
% [0  1   0   0]
% [0  0   1   0]
% [tx ty  tz  1]

% Rotation x:
% [1     0       0      0]
% [0  cos(a)   sin(a)   0]
% [0  -sin(a)  cos(a)   0]
% [0     0        0     1]

% Rotation y: 
% [cos(a)  0  -sin(a)   0]
% [0       1     0      0]
% [sin(a)  0   cos(a)   0]
% [0       0     0      1]

% Rotation z:
% [ cos(a)  sin(a)  0   0]
% [-sin(a)  cos(a)  0   0]
% [ 0         0     1   0]
% [ 0         0     0   1]

% imrotate3 !!
% imtranslate !!

fixedVolume = single(V_pre);
movingVolume = single(V_post_adj);

% test gpu optimization -- not working!!!
fixedgpu = rgb2gray(gpuArray(fixedVolume));
movinggpu = rgb2gray(gpuArray(movingVolume));

[optimizer,metric] = imregconfig('monomodal');
% movingRegisteredVolume = imregister(movingVolume, fixedVolume, 'rigid', optimizer, metric);

geomtform = imregtform(movingVolume,fixedVolume, 'rigid', optimizer, metric,'InitialTransformation',geomtform)
geomtform.T

% [~,movingReg] = imregdemons(movingGPU,fixedGPU,[500 400 200],'AccumulatedFieldSmoothing',1.3);
% movingReg = gather(movingReg);

movingRegisteredVolume = imwarp(movingVolume,geomtform,'bicubic');

figure,imshowpair(movingRegisteredVolume(:,:,220), fixedVolume(:,:,220));
title('Axial Slice of Registered Volume')
figure,imshowpair(movingRegisteredVolume, fixedVolume, 'montage');

helperVolumeRegistration(fixedVolume,movingRegisteredVolume);


% centerXWorld = mean(Rmoving.XWorldLimits);
% centerYWorld = mean(Rmoving.YWorldLimits);
% centerZWorld = mean(Rmoving.ZWorldLimits);
% [xWorld,yWorld,zWorld] = transformPointsForward(geomtform,centerXWorld,centerYWorld,centerZWorld);
% 
% [r,c,p] = worldToSubscript(Rfixed,xWorld,yWorld,zWorld)


%% Making manual changes
 
% % imrotate3 
% % imtranslate -- just use imwarp with correct transformation matrix
% 
% movingRegisteredVolume_2 = movingRegisteredVolume;
% % [V1_bone,V2_bone,V_tog] = combine_bone(fixedVolume_whole,movingRegisteredVolume_2);
%     
% rot = [0,0,0,0];
% trans =  [0,0,0];
% tmat = [[1,0,0,0];[0,1,0,0];[0,0,1,0];[trans(1),trans(2),trans(3),1]];
% tform_temp = affine3d(tmat);
% while 1
%     
%     % figure,imshowpair(movingRegisteredVolume_2(:,:,220), fixedVolume(:,:,220));
%     % title('Axial Slice of Registered Volume')
%     % figure,imshowpair(movingRegisteredVolume_2, fixedVolume, 'montage');
%     % 
%     % helperVolumeRegistration(fixedVolume,movingRegisteredVolume_2);
% 
%     
%     
%     str = input('Enter movement type: ','s');
%     
%     if str(1) == 'Q' || str(1) == 'q'
%         break
%     elseif str(1) == 'T' || str(1) == 't'
%         trans(1) = input('Enter tx: ');
%         trans(2) = input('Enter ty: ');
%         trans(3) = input('Enter tz: ');
%         tmat = [[1,0,0,0];[0,1,0,0];[0,0,1,0];[trans(1),trans(2),trans(3),1]];
%         tform_temp = affine3d(tmat);
%     elseif str(1) == 'R' || str(1) == 'r'
%         rot(1) = input('Enter angle: ');
%         rot(2) = input('Enter xaxis: ');
%         rot(3) = input('Enter yaxis: ');
%         rot(4) = input('Enter zaxis: ');
%     else
%         disp("Invalid input")
%         continue
%     end
%     
%     movingRegisteredVolume_2 = imwarp(movingRegisteredVolume,Rmoving_whole,tform_temp,'bicubic','OutputView',Rfixed_whole);
%     movingRegisteredVolume_2 = imrotate3(movingRegisteredVolume_2,rot(1),rot(2:4));
%     
%     [V1_bone,V2_bone,V_tog] = combine_bone(fixedVolume_whole,movingRegisteredVolume_2);
%     
% end
% % volshow(V1_bone)
% % volshow(V2_bone)
% % diffdisplay_slice(V_tog)
% 

%% Volume using points (using lines) - sloped height - one jaw only
V_bottom_together = bottomjaw_points(V_together);

% CHECK - if displayed 1,2,251: correct
for i = 1:256
    if sum(sum(sum(V_bottom_together==i))) ~= 0
        disp(i)
    end
end

% Calculating volume for bottom jaw
temp_inter = sum(sum(sum(V_bottom_together==1)));
temp_pre = sum(sum(sum(V_bottom_together==5)));
temp_post = sum(sum(sum(V_bottom_together==250)));

d_bottom_before = temp_inter + temp_pre
d_bottom_after = temp_inter + temp_post
d_bottom_difference = d_bottom_after - d_bottom_before


%% Volume using points (using lines) - sloped height - one jaw only
V_top_together = topjaw_points(V_together);

% CHECK - if displayed 1,2,251: correct
for i = 1:256
    if sum(sum(sum(V_top_together==i))) ~= 0
        disp(i)
    end
end

% Calculating volume for bottom jaw
temp_inter = sum(sum(sum(V_top_together==1)));
temp_pre = sum(sum(sum(V_top_together==5)));
temp_post = sum(sum(sum(V_top_together==250)));

d_top_before = temp_inter + temp_pre
d_top_after = temp_inter + temp_post
d_top_difference = d_top_after - d_top_before

%% Volume using points (using surfaces) - one jaw only - too slow so no

V_bottom_jaw = V_all_bone;
for i = 1:557
    for j = 1:557
        for k = 1:441
            isPart = ((j-point1(2))*m1d>m1n*(i-point1(1))) & ...
                 ((j-point2(2))*m2d>m2n*(i-point2(1))) & ...
                 ((j-point3(2))*m3d>m3n*(i-point3(1))) & ...
                 ((j-point4(2))*m4d>m4n*(i-point4(1))) & ...
                 ((k-point1(3))*m5d>m5n*(i-point1(1))) & ...
                 ((k-point5(3))*m6d<m6n*(i-point5(1)));
            V_bottom_jaw(i,j,:) = V_bottom_jaw(i,j,:)*isPart;
        end
    end
end
volshow(V_bottom_jaw)


%% Finding jaw from image using orthogonal planes

% V_half_bone = V_all_bone(:,1:300,200:441); %
% V_top_jaw = V_all_bone(:,:,180:441);
% V_bottom_jaw = V_all_bone(:,:,1:220);
% V_nose = V_top_jaw(50:100,:,1:150);
% % V_nose(:,250:300,:) = 0;
% Image_nose = V_all_bone(100,:,:);
% % figure; volshow(V_top_jaw)
% % figure; volshow(V_bottom_jaw)
% figure; volshow(V_all_bone)
% % plot(Image_nose)
% % need to smoothen the surface of nose
% % need to take minima of nasal cavity - formas upper edge
% % 
% 
% %%smoothening through nonlinear regression
% X = [Horsepower,Weight];
% yval = MPG;
% 
% modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + ...
%     b(4)*x(:,2).^b(5);
% beta0 = [-50 500 -1 500 -1];
% mdl = fitnlm(X,yval,modelfun,beta0)
% 
% Xnew = nanmean(X)  
% 
% MPGnew = predict(mdl,Xnew)


% STARTING AGAIN
volcalculate(V_together)



%% Finding jaw from image using points at trapezoid edges

% need to first be able to mark points on diagram 

% CANNOT DO

%% examples

% I = imread('toyobjects.png');
% imshow(I)
%   
% volshow(V_all_bone)
% mask = roipoly;
%   
% figure, imshow(mask)
% title('Initial MASK');

V3 = mat2gray(V);
cmap = copper(256);
% reV = reshape(V3,557,557,441,1);
% reV = ind2rgb(reV, cmap);

numslice = size(V3,3);
reV = zeros(size(V3, 1), size(V3, 2), 3, numslice);
for slice = 1 : numslice
  reV(:,:,:,slice) = ind2rgb(V3(:,:,slice), cmap);
end

handle.a = axes;
% generate random x,y,z values
handle.x = randi([1 10],1,5);
handle.y = randi([1 10],1,5);
handle.z = randi([1 10],1,5);
% plot in 3D
handle.p = plot3(handle.x,handle.y,handle.z,'.');
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');
% add callback when point on plot object 'handle.p' is selected
% 'click' is the callback function being called when user clicks a point on plot
handle.p.ButtonDownFcn = {@click,handle};
% definition of click



%% Finding jaw from image using image processing

% try displaying gradient to see if helpful
% activecontour	Segment image into foreground and background using active contours (snakes)
% bfscore	Contour matching score for image segmentation
% dice	Sørensen-Dice similarity coefficient for image segmentation
% gradientweight	Calculate weights for image pixels based on image gradient
% graydiffweight	Calculate weights for image pixels based on grayscale intensity difference
% imsegfmm	Binary image segmentation using Fast Marching Method
% imsegkmeans3	K-means clustering based volume segmentation
% jaccard	Jaccard similarity coefficient for image segmentation
% superpixels3	3-D superpixel oversegmentation of 3-D image
% Finding surface of image (is useful?)

% doubtful if possible


%% Trying image segmentation (unclear results)

%update threshold for binary imaging
%change slices, can use more slices to help?
%test with diffferent types of active contour
%test with diffferent repetitions of active contour

z_slice = 160;
y_slice = 75;
XY = Vsingle(:,:,z_slice);
XZ = squeeze(Vsingle(y_slice,:,:));
% figure, imshow(XY, [],'Border','tight');
% figure, imshow(XZ, [],'Border','tight');

% imageSegmenter(XY) %interactive way to create 2d mask

%clearing XY sliced image
BW = imbinarize(XY,0.019608); %not good, set manual threshold and do
BW = imclearborder(BW);
BW = imfill(BW, 'holes');
radius = 3;
decomposition = 0;
se = strel('disk',radius,decomposition);
BW = imerode(BW, se);
BW = imdilate(BW, se);
BW_XY = BW;
maskedImageXY = XY;
maskedImageXY(~BW) = 0;
figure,imshow(maskedImageXY)

%clearing XZ sliced image
BW = imbinarize(XZ,0.019608); %not good, set manual threshold and do
BW = imclearborder(BW);
BW = imfill(BW,'holes');
radius = 13;
decomposition = 0;
se = strel('disk',radius,decomposition);
BW = imerode(BW, se);
BW = imdilate(BW, se);
BW_XZ = BW;
maskedImageXZ = XZ;
maskedImageXZ(~BW) = 0;
figure,imshow(maskedImageXZ)

jaw_front = 1;
jaw_back = 320;

jaw_bottom=50;
jaw_top=320;

%creating minimal 3d mask
mask = false(size(V));
mask(jaw_front:jaw_back,:, z_slice) = maskedImageXY(jaw_front:jaw_back,:,:);
mask(y_slice, :, :) = mask(y_slice, :, :)|reshape(maskedImageXZ, [1, 557, 441]);

% creating maximum 3d mask
mask2 = false(size(V));
mask2(jaw_front:jaw_back,:,jaw_bottom:jaw_top) = true;

%active contour
V_hist = histeq(Vsingle).*V_all_bone;

% %test 1, same as lung example
% BW  = activecontour(V_hist,mask,100,'edge'); %see required properties
% segmentedImage = V_hist.*single(BW);
% 
% %test 2, with higher iterations
% BW  = activecontour(V_hist,mask,100,'edge'); %see required properties
% segmentedImage = V_hist.*single(BW);
% 
% %test 3, with smoothness factor high
% BW  = activecontour(V_hist,mask,100,'edge'); %see required properties
% segmentedImage = V_hist.*single(BW);

%test 4, edge with shrinking
BW  = activecontour(V_hist,mask2,10,'edge','SmoothFactor',0,'ContractionBias',1); %see required properties
segmentedImage = V_hist.*single(BW);



%view segment
volumeViewer(segmentedImage)

% %volume of segment
% volLungsPixels = regionprops3(logical(BW),'volume');
% 
% %Calculating of x,y,z dimensions
% volLungs1 = volLungsPixels.Volume(1)*0.76*0.76*1.25*1e-6;
% volLungs2 = volLungsPixels.Volume(2)*0.76*0.76*1.25*1e-6;
% volLungsLiters = volLungs1 + volLungs2

%% Viewing possiblities to display 3d image
% input grayscale image V or binary image V_all_bone of size 557x557x441
% viewer3d(V_all_bone) --- not showing output?
% Windowlevel(V) ---- warning: hangs 
% vol3d('CData', V); ---- hangs? need rgb image?
% Volumetric  ----- exactly what required!!! but how to use!? SPM...
% plot3(V) --- need X,Y,X inputs, how to find? Can only plot for single
% z/ surface
% V_half_bone = V_all_bone(:,1:300,200:441);
% x = [];
% y = [];
% z = [];
% for i = 1:557
%     for j = 1:300
%         for k = 1:242
%             if V_half_bone(i,j,k) == 1
%                 x = [x i];
%                 y = [y j];
%                 z = [z k];
%             end
%         end
%     end
% end
% plot3(x,y,z)
% sliceomatic ----- is good worst case, slow, slices
% V2 = im2double(V);
% sliceomatic(V2)


%% Volume using points (using lines) -fixed height - one jaw only

% point 1,2,3,4 top; 5,6,7,8 bottom; clockwise from top

%point = [x,y,z]; x-front to back,y-left to right(looking at face),z-bottom to top
point1 = [1,400,50];
point2 = [320,1,point1(3)];
point3 = [320,300,point1(3)];
point4 = [1,557,point1(3)];
point5 = [point1(1:2),320];
point6 = [point2(1:2),point5(3)];
point7 = [point3(1:2),point5(3)];
point8 = [point4(1:2),point5(3)];

point1 = [1,557,1];
point2 = [320,1,1];
point3 = [320,1,point1(3)];
point4 = [1,557,point2(3)];
point5 = [point1(1:2),441];
point6 = [point2(1:2),441];
point7 = [point3(1:2),point5(3)];
point8 = [point4(1:2),point6(3)];

% finding volume assuming same trapezium size for top and bottom on parallel planes

height = [point1(3),point5(3)];

% creating lines for each side of trapezium (equation of y wrt x)
m1n = (point2(2)-point1(2));
m1d = (point2(1)-point1(1));
% c1 = point2(2)-(m1*point2(1));
m2n = (point3(2)-point2(2));
m2d = (point3(1)-point2(1));
% c2 = point3(2)-(m2*point3(1));
m3n = (point4(2)-point3(2));
m3d = (point4(1)-point3(1));
% c3 = point4(2)-(m3*point4(1));
m4n = (point1(2)-point4(2));
m4d = (point1(1)-point4(1));
% c4 = point1(2)-(m4*point1(1));

V_bottom_jaw = V_all_bone(:,:,height(1):height(2));
for i = 1:557
    for j = 1:557
        isPart = ((j-point1(2))*m1d>m1n*(i-point1(1))) & ...
                 ((j-point2(2))*m2d>m2n*(i-point2(1))) & ...
                 ((j-point3(2))*m3d>m3n*(i-point3(1))) & ...
                 ((j-point4(2))*m4d>m4n*(i-point4(1)));
        V_bottom_jaw(i,j,:) = V_bottom_jaw(i,j,:)*isPart;
    end
end

diffdisplay(V_bottom_jaw)

%% Random



% Display side by side for each slice to analyze
for i= 1:441
subplot(1,2,1)
imshow(Vsingle(:,:,i))
subplot(1,2,2)
imshow(movingRegisteredVolume(:,:,i))
pause
end

% Once in Directory
for i = 1:441
metadata1 = dicominfo(strcat('3DSlice',string(i),'.dcm'));
dicomwrite(movingRegisteredVolume_int(:,:,i),strcat('3DSlice',string(i),'.dcm'),metadata1);
end

% Hist equalisation
nbins = 65535;
V_post_2 = imhistmatchn(uint16(movingRegisteredVolume),V,nbins);



%% Calculating threshold value based on histogram

vhist1 = histogram(V_pre);
vhist2 = hist(V_post_raw);
plot(vhist1);
plot(vhist2);


%% Functions

function [histo] = calchist(V)
    % return 1d array of histogram for 3d image
    bottom = min(min(min(V)));
    top = max(max(max(V)));
    histo = zeros(1, top-bottom+1);
    for i = bottom:top
        histo(i-bottom+1) = sum(sum(sum(V==i)));
    end

end

function [] = volcalculate(V)
    jaw_front = 1;
    jaw_back = 250;

    jaw_bottom=50;
    jaw_top=320;

    %creating minimal 3d mask
    mask2 = false(size(V));
    mask2(jaw_front:jaw_back,:,jaw_bottom:jaw_top) = true;

    V_jaw2 = V.*mask2;
%     volshow(V_jaw2)
    diffdisplay(V);

%     radius = 10;
%     decomposition = 0;
%     se = strel('sphere',radius);
%     mask3 = imopen(V_jaw2,se);
%     V_jaw3 = V.*mask3;
% 
%     h = volshow(V_jaw3);
end

function [V_bottom_jaw] = bottomjaw_points(V)
    % Volume using points (using lines) - sloped height - one jaw only

    % point 1,2,3,4 top; 5,6,7,8 bottom; clockwise from top

    jaw_front = 1;
    jaw_back = 250;

    jaw_bottom=50;
    jaw_top=320;

    %point = [x,y,z]; x-front to back,y-left to right(looking at face),z-bottom to top
    point1 = [1,1,50];
    point2 = [220,1,50];
    point3 = [220,557,point2(3)];
    point4 = [1,557,point1(3)];
    point5 = [point1(1:2),165];
    point6 = [point2(1:2),235];
    point7 = [point3(1:2),point6(3)];
    point8 = [point4(1:2),point5(3)];

    % finding volume assuming same trapezium size for top and bottom on 
    % sloped planes (perpendicular to y axis)

    % creating lines for each side of trapezium (equation of y wrt x)
    m1n = (point2(2)-point1(2));
    m1d = (point2(1)-point1(1));
    m2n = (point3(2)-point2(2));
    m2d = (point3(1)-point2(1));
    m3n = (point4(2)-point3(2));
    m3d = (point4(1)-point3(1));
    m4n = (point1(2)-point4(2));
    m4d = (point1(1)-point4(1));

    % creating lines for each side of trapezium (equation of z wrt x)
    m5n = (point2(3)-point1(3));
    m5d = (point2(1)-point1(1));
    m6n = (point6(3)-point5(3));
    m6d = (point6(1)-point5(1));


    V_bottom_jaw = V;
    for i = 1:557
        for j = 1:557
            isPart = ((j-point1(2))*m1d>m1n*(i-point1(1))) & ...
                     ((j-point2(2))*m2d>m2n*(i-point2(1))) & ...
                     ((j-point3(2))*m3d>m3n*(i-point3(1))) & ...
                     ((j-point4(2))*m4d>m4n*(i-point4(1)));
            V_bottom_jaw(i,j,:) = V_bottom_jaw(i,j,:)*isPart;
        end
    end

    % volshow(V_bottom_jaw)
    % pause;

    for i = 1:557
        for k = 1:441
            isPart = ((k-point1(3))*m5d>m5n*(i-point1(1))) & ...
                     ((k-point5(3))*m6d<m6n*(i-point5(1))); %check this!!!
            V_bottom_jaw(i,:,k) = V_bottom_jaw(i,:,k)*isPart;
        end
    end

    diffdisplay(V_bottom_jaw)
end

function [V_top_jaw] = topjaw_points(V)
    % Volume using points (using lines) - sloped height - one jaw only

    % point 1,2,3,4 top; 5,6,7,8 bottom; clockwise from top

    jaw_front = 1;
    jaw_back = 250;

    jaw_bottom=50;
    jaw_top=320;

    %point = [x,y,z]; x-front to back,y-left to right(looking at face),z-bottom to top
    point1 = [1,1,165];
    point2 = [220,1,235];
    point3 = [220,557,point2(3)];
    point4 = [1,557,point1(3)];
    point5 = [point1(1:2),320];
    point6 = [point2(1:2),320];
    point7 = [point3(1:2),point6(3)];
    point8 = [point4(1:2),point5(3)];

    % finding volume assuming same trapezium size for top and bottom on 
    % sloped planes (perpendicular to y axis)

    % creating lines for each side of trapezium (equation of y wrt x)
    m1n = (point2(2)-point1(2));
    m1d = (point2(1)-point1(1));
    m2n = (point3(2)-point2(2));
    m2d = (point3(1)-point2(1));
    m3n = (point4(2)-point3(2));
    m3d = (point4(1)-point3(1));
    m4n = (point1(2)-point4(2));
    m4d = (point1(1)-point4(1));

    % creating lines for each side of trapezium (equation of z wrt x)
    m5n = (point2(3)-point1(3));
    m5d = (point2(1)-point1(1));
    m6n = (point6(3)-point5(3));
    m6d = (point6(1)-point5(1));


    V_top_jaw = V;
    for i = 1:557
        for j = 1:557
            isPart = ((j-point1(2))*m1d>m1n*(i-point1(1))) & ...
                     ((j-point2(2))*m2d>m2n*(i-point2(1))) & ...
                     ((j-point3(2))*m3d>m3n*(i-point3(1))) & ...
                     ((j-point4(2))*m4d>m4n*(i-point4(1)));
            V_top_jaw(i,j,:) = V_top_jaw(i,j,:)*isPart;
        end
    end

    for i = 1:557
        for k = 1:441
            isPart = ((k-point1(3))*m5d>m5n*(i-point1(1))) & ...
                     ((k-point5(3))*m6d<m6n*(i-point5(1))); %check this!!!
            V_top_jaw(i,:,k) = V_top_jaw(i,:,k)*isPart;
        end
    end

    diffdisplay(V_top_jaw)
end

function [] = diffdisplay(V)
    Transparency = reshape(linspace(1,1,256),256,1);
    Transparency(1) = 0;
    Colouring = winter(256);
    for i = 1:100
        Colouring(i,:) = [1,1,1]; %white
    end
    for i = 101:256
        Colouring(i,:) = [255/256,105/256,180/256]; %pink
    end
    figure; volshow(V,'Alphamap',Transparency,'Colormap',Colouring)
end

function [V_bone_pre,V_bone_post] = BoneVolume(V_pre,V_post,Bone_cutoff)
% separating bone by (V_pre,V_post)

    % Bone_cutoff = 1200;
    V_bone_pre = V_pre > Bone_cutoff;
    Num_elements = sum(sum(sum(V_pre(200:400,:,:)>Bone_cutoff)));
    for i = Bone_cutoff:1:-1
        if Num_elements < sum(sum(sum(V_post(200:400,:,:)>i)))
            break
        end
    end
    V_bone_post = V_post > i;
end

function [fused] = Overlay_images(V1,V2)
% Comparing 2 3d images (need to test with 2 different images)
% Inputs are V1 and V2 of dim 557,557,441

fused = zeros(557,557,441,3);
for i = 1:441
    fused(:,:,i,:) = imfuse(V1(:,:,i),V2(:,:,i));
end

imshow3D(fused) %gives slice by slice image, see if any other way to view rgb image
end

function [volume] = Numerical_volume(V_bin)
% Finding volume of selected jaw (or any binary image). 
% Input is 3d binary image of jaw V_bin

volume_array = regionprops3(V_bin,'Volume');
volume = sum(volume_array.Volume);
end

function click(obj,eventData,handle)
    % co-ordinates of the current selected point
    Pt = handle.a.CurrentPoint(2,:);
    % find point closest to selected point on the plot
    for k = 1:5
        arr = [handle.x(k) handle.y(k) handle.z(k);Pt];
        distArr(k) = pdist(arr,'euclidean');
    end
    [~,idx] = min(distArr);
    % display the selected point on plot
    disp([handle.x(idx) handle.y(idx) handle.z(idx)]);
end
