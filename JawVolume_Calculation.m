%% Creating 3d Volume and bone data
%inputs: Bone_cutoff, Image to read name/path

% Remaining: Soujanya (values), Ashakiran, Hitaishi
patient_name = 'SOUJANYA';
% Input variables
V_pre_name = strcat('VOL_',patient_name,'_BEFORE');
V_post_name = strcat('VOL_',patient_name,'_AFTER');

% Reading image details
V_pre_info = dicom_folder_info(V_pre_name);
V_post_info = dicom_folder_info(V_post_name);

% Reading image
V_pre = squeeze(dicomreadVolume(V_pre_name));
V_post_raw = squeeze(dicomreadVolume(V_post_name));

dim_post = size(V_post_raw);
V_post_raw = V_pre;
V_post_raw(1:dim_post(1),1:dim_post(2),1:dim_post(3))=squeeze(dicomreadVolume(V_post_name));

%% Histogram equalisation / preprocessing

V_post_adj = fix_intensity(V_post_raw,V_pre);

%% View overlap before processing

% [V1_bone,V2_bone,V_tog] = combine_bone(V_pre,V_post_adj);


%% Superimpose images 
                        
% how to store header information? need struct with 
% SliceThickness: _  --- in DICOM header, 
% PixelSize: [_,_]   --- in DICOM header, PixelSpacings (0.3,0.3)

fixedVolume = single(V_pre);
movingVolume = single(V_post_adj);

[optimizer,metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 500;
% optimizer.MaximumStepLength = optimizer.MaximumStepLength * 3;
optimizer.MinimumStepLength = optimizer.MinimumStepLength / 10;

Rfixed   = imref3d(size( fixedVolume), V_pre_info.Scales(1), V_pre_info.Scales(2), V_pre_info.Scales(3));
Rmoving  = imref3d(size(movingVolume),V_post_info.Scales(1),V_post_info.Scales(2),V_post_info.Scales(3));

% movingRegisteredVolume = imregister(movingVolume,Rmoving,fixedVolume,Rfixed,'rigid',optimizer,metric);

geomtform = imregtform(movingVolume,Rmoving,fixedVolume,Rfixed,'rigid',optimizer,metric)
geomtform.T


movingRegisteredVolume = imwarp(movingVolume,Rmoving,geomtform,'bicubic','OutputView',Rfixed);

figure,imshowpair(movingRegisteredVolume(:,:,220), fixedVolume(:,:,220));
title('Axial Slice of Registered Volume')


%% Testing quality of registration

% [V1_bone,V2_bone,V_tog] = combine_bone(fixedVolume,movingRegisteredVolume);


%% Assign Registered image

V_post = uint16(movingRegisteredVolume);

%% Bone Volumes

% Input variables
Bone_cutoff = 1200;

% Creating binary images of bone
[V_bone_pre,V_bone_post,V_together] = combine_bone(V_pre,V_post,Bone_cutoff);


%% Run this test to see if jaw edges are correct (must run)

% input values for bottom jaw
bottom_jaw_front = 1; %try 1
bottom_jaw_back = 190; %try 190

bottom_jaw_left = 1; %try 1
bottom_jaw_right = 557; %try 557

bottom_jaw_bottom1=5; %try 5
bottom_jaw_bottom2=80; %try 80
bottom_jaw_top1=180; %try 180
bottom_jaw_top2=200; %try 200

% input values for top jaw
top_jaw_front = bottom_jaw_front; %try 1
top_jaw_back = bottom_jaw_back; %try 190

top_jaw_left = bottom_jaw_left; %try 1
top_jaw_right = bottom_jaw_right; %try 557

top_jaw_bottom1=bottom_jaw_top1; %try 180
top_jaw_bottom2=bottom_jaw_top2; %try 200
top_jaw_top1=270; %try 270
top_jaw_top2=top_jaw_top1; %try 270

num_dim = 3; % between 1 and 6
% 1 - bottom 
% 2 - top 
% 3 - back
% 4 - front
% 5 - left
% 6 - right

% displaying values for bottom jaw
jaw_points(V_together,bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,num_dim);

% displaying values for top jaw
jaw_points(V_together,top_jaw_front,...
    top_jaw_back,top_jaw_left,top_jaw_right,top_jaw_top1,...
    top_jaw_top2,top_jaw_bottom1,top_jaw_bottom2,num_dim);


%% Calculate volume of jaws -- processing
% BOTTOM JAW

mask_pre = jaw_points(V_bone_pre,bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,0);

mask_post = jaw_points(V_bone_post,bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,0);

% Processing images to remove noise and fill holes
% mask_pre = process_mask(mask_pre);
% mask_post = process_mask(mask_post);
% mask = mask_pre | mask_post;

V_pre_bottom = V_pre .* uint16(mask_pre);
V_post_bottom = V_post .* uint16(mask_post);

[optimizer_bottom,metric] = imregconfig('monomodal');
optimizer_bottom.MaximumIterations = 500;
% optimizer_bottom.MaximumStepLength = optimizer_botto  m.MaximumStepLength * 3;
optimizer_bottom.MinimumStepLength = optimizer_bottom.MinimumStepLength / 10;

Rfixed_jaw   = imref3d(size( V_pre_bottom), V_pre_info.Scales(1), V_pre_info.Scales(2), V_pre_info.Scales(3));
Rmoving_jaw  = imref3d(size(V_post_bottom),V_post_info.Scales(1),V_post_info.Scales(2),V_post_info.Scales(3));

V_post_bottom_alt = imregister(V_post_bottom,Rmoving_jaw,V_pre_bottom,Rfixed_jaw,'rigid',optimizer_bottom,metric);

% geomtform_jaw = imregtform(V_post_bottom,Rmoving_jaw,V_pre_bottom,Rfixed_jaw,'rigid',optimizer_bottom,metric)
% geomtform_jaw.T
% 
% V_post_bottom_alt = imwarp(V_post_bottom,Rmoving_jaw,geomtform_jaw,'bicubic','OutputView',Rfixed_jaw);
V_post_bottom_alt_adj = fix_intensity(V_post_bottom_alt,V_pre_bottom);
% V_post_bottom_alt_adj = imfill(V_post_bottom_alt_adj, 'holes');

figure,imshowpair(V_post_bottom_alt_adj(:,:,150), V_pre_bottom(:,:,150));
title('Axial Slice of Registered Volume Bottom Jaw')



%% Calculate volume of jaws -- processing
% TOP JAW

mask_pre = jaw_points(V_bone_pre,top_jaw_front,...
    top_jaw_back+30,top_jaw_left,top_jaw_right,top_jaw_top1+30,...
    top_jaw_top2+30,top_jaw_bottom1-30,top_jaw_bottom2-30,0);

mask_post = jaw_points(V_bone_post,top_jaw_front,...
    top_jaw_back+30,top_jaw_left,top_jaw_right,top_jaw_top1+30,...
    top_jaw_top2+30,top_jaw_bottom1-30,top_jaw_bottom2-30,0);

% Processing images to remove noise and fill holes
% kk!!!

V_pre_top = V_pre .* uint16(mask_pre);
V_post_top = V_post .* uint16(mask_post);

[optimizer_top,metric] = imregconfig('monomodal');
optimizer_top.MaximumIterations = 500;
% optimizer_top.MaximumStepLength = optimizer_top.MaximumStepLength * 3;
optimizer_top.MinimumStepLength = optimizer_top.MinimumStepLength / 10;

Rfixed_jaw   = imref3d(size( V_pre_top), V_pre_info.Scales(1), V_pre_info.Scales(2), V_pre_info.Scales(3));
Rmoving_jaw  = imref3d(size(V_post_top),V_post_info.Scales(1),V_post_info.Scales(2),V_post_info.Scales(3));

V_post_top_alt = imregister(V_post_top,Rmoving_jaw,V_pre_top,Rfixed_jaw,'rigid',optimizer_top,metric);

% geomtform_jaw = imregtform(V_post_top,Rmoving_jaw,V_pre_top,Rfixed_jaw,'rigid',optimizer_top,metric)
% geomtform_jaw.T
% 
% V_post_top_alt = imwarp(V_post_top,Rmoving_jaw,geomtform_jaw,'bicubic','OutputView',Rfixed_jaw);
V_post_top_alt_adj = fix_intensity(V_post_top_alt,V_pre_top);

figure,imshowpair(V_post_top_alt_adj(:,:,250), V_pre_top(:,:,250));
title('Axial Slice of Registered Volume Top Jaw')




%% Calculate volume of jaws -- calculation

V1 = uint16(jaw_points(single(V_pre),bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,0));

V2 = uint16(jaw_points(single(V_post),bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,0));

% V1a = imadjustn(V1);

% V2a = imadjustn(V2);%,[double(0),max(max(max(V2)))],[double(0),max(max(max(V1)))]);

% V3 = uint16(jaw_points(single(V_pre),top_jaw_front,...
%     top_jaw_back+30,top_jaw_left,top_jaw_right,top_jaw_top1+30,...
%     top_jaw_top2+30,top_jaw_bottom1-30,top_jaw_bottom2-30,0));

% V4 = uint16(jaw_points(single(V_post),top_jaw_front,...
%     top_jaw_back+30,top_jaw_left,top_jaw_right,top_jaw_top1+30,...
%     top_jaw_top2+30,top_jaw_bottom1-30,top_jaw_bottom2-30,0));

% V3a = imadjustn(V3);

% V4a = imadjustn(V4);%,[double(0),double(max(max(max(V4))))],[double(0),double(max(max(max(V3))))]);

% V22 = fix_intensity(V2,V1);
% V11 = fix_intensity(V1,V2);

% V44 = fix_intensity(V4,V3);
% V33 = fix_intensity(V3,V4a);

% BOTTOM JAW

% Cropping to bottom jaw
[~,~,V_bottom_together] = combine_bone(V_pre_bottom,V_post_bottom_alt_adj,10); 
% temp = Bone_cutoff*(max(max(max(V1)))/max(max(max(V_pre))));
% [~,~,V_bottom_together] = combine_bone(V1,V2,temp); 
V_bottom_together = jaw_points(V_bottom_together,bottom_jaw_front,...
    bottom_jaw_back,bottom_jaw_left,bottom_jaw_right,bottom_jaw_top1,...
    bottom_jaw_top2,bottom_jaw_bottom1,bottom_jaw_bottom2,0);

% arr for values
arr = [];
for i = 1:256
    if sum(sum(sum(V_bottom_together==i))) ~= 0
        arr = [arr i];
    end
end

if (size(arr)~=3)
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    return
end

% Calculating volume for bottom jaw
temp_inter = Numerical_volume(V_bottom_together==arr(1));
temp_pre = Numerical_volume(V_bottom_together==arr(2));
temp_post = Numerical_volume(V_bottom_together==arr(3));
%
d_bottom_before = temp_inter + temp_pre
d_bottom_after = temp_inter + temp_post
d_bottom_difference = d_bottom_after - d_bottom_before
d_bottom_difference_mm = d_bottom_difference * 0.027

% TOP JAW

% Cropping to top jaw

[~,~,V_top_together] = combine_bone(V_pre_top,V_post_top_alt_adj,10); 
% temp = Bone_cutoff*(max(max(max(V3)))/max(max(max(V_pre))));
% [~,~,V_top_together] = combine_bone(V3,V4,temp); 
V_top_together = jaw_points(V_top_together,top_jaw_front,...
    top_jaw_back,top_jaw_left,top_jaw_right,top_jaw_top1,...
    top_jaw_top2,top_jaw_bottom1,top_jaw_bottom2,0);

% arr for values
arr = [];
for i = 1:256
    if sum(sum(sum(V_top_together==i))) ~= 0
        arr = [arr i];
    end
end

if (size(arr)~=3)
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    disp("ERROR!!! PLEASE CONTACT SNEHA THANK YOU")
    return
end

% Calculating volume for top jaw
temp_inter = Numerical_volume(V_top_together==arr(1));
temp_pre = Numerical_volume(V_top_together==arr(2));
temp_post = Numerical_volume(V_top_together==arr(3));

d_top_before = temp_inter + temp_pre
d_top_after = temp_inter + temp_post
d_top_difference = d_top_after - d_top_before
d_top_difference_mm = d_top_difference * 0.027


%% Saving results

file_name = strcat('VREG_',patient_name,'.mat');
save(file_name,'V_pre','V_pre_top','V_pre_bottom','V_post','V_post_bottom_alt',...
    'V_post_bottom_alt_adj','V_post_top_alt','V_post_top_alt_adj',...
    'd_bottom_difference_mm','d_top_difference_mm','geomtform');


%% Option to view combination of any 2 volumes

% [V1_bone,V2_bone,V_tog] = combine_bone(V_pre,V_post_adj);

% display_each_slice(V_pre_bottom,V_post_bottom_alt_adj);

% volshow(V1_bone)
% volshow(V2_bone)

% diffdisplay(V_tog)
% diffdisplay_slice(V_top_together)


%% Calculating volume using boundary

% V3 = V1;
% for i = 100:101
%     I = V1(:,:,i);
%     imshow(I)
%     BW = I>1200;
%     BW_filled = imfill(BW,'holes');
%     imshow(BW)
%     dim = size(BW);
%     col = round(dim(2)/2)-90;
%     row = min(find(BW(:,col)));
%     if row
        
%     else 
%         continue
%     end
%     boundary = bwtraceboundary(BW,[row, col],'N');
%     imshow(I)
%     plot(boundary(:,2),boundary(:,1),'g','LineWidth',3);
%     I2 = zeros(557,557);
%     for j = 1:length(boundary)
%         I2(boundary(j,1),boundary(j,2)) = 1;
%     end
%     V3(:,:,i) = I2;
% end




%% Functions


function [] = display_each_slice(V1,V2)
% Display side by side for each slice to analyze
    figure;
    for i= 1:441
        imshowpair(V1(:,:,i),V2(:,:,i))
        pause
    end
end

function [] = diffdisplay_slice(V)
    Transparency = reshape(linspace(1,1,256),256,1);
    Transparency(1) = 0;
    Colouring = winter(256);
    for i = 1:30
        Colouring(i,:) = [1,1,1]; %white
    end
    for i = 31:100
        Colouring(i,:) = [173/256,255/256,47/256]; %green
    end
    for i = 101:256
        Colouring(i,:) = [255/256,105/256,180/256]; %pink
    end
    figure; 
    for i = 1:441
        imshow(V(:,:,i),'Colormap',Colouring)
        pause
    end
    volshow(V,'Alphamap',Transparency,'Colormap',Colouring);
end

function [] = store_dicom(V,dir)
% store dicom image to folder - not working currently - patient orientation?
    olddir = cd(dir);
    for i = 1:441
        slice_file = strcat('3DSlice',string(i),'.dcm');
        metadata1 = dicominfo(slice_file);
        dicomwrite(V(:,:,i),slice_file,metadata1);
    end
    cd olddir
end

function [V_post_2] = fix_intensity(V1,V2)
    % Hist equalisation
    nbins = 65535;
    V_post_2 = imhistmatchn(V1,V2,nbins);
end


function [V_bone_pre,V_bone_post] = BoneVolume(V_pre,V_post,Bone_cutoff)
% separating bone by (V_pre,V_post)

    % Bone_cutoff = 1200;
    V_bone_pre = V_pre > Bone_cutoff;
%     Num_elements = sum(sum(sum(V_pre(200:400,:,60:380)>Bone_cutoff)));
%     for i = Bone_cutoff:-1:1
%         if Num_elements < sum(sum(sum(V_post(200:400,:,60:380)>i)))
%             break
%         end
%     end
%     V_bone_post = V_post > i;
    V_bone_post = V_post > (Bone_cutoff);
end

function [V1_bone,V2_bone,V_together] = combine_bone(V1,V2,Bone_cutoff)
    if nargin < 3
        Bone_cutoff = 1200;
    end
    % Creating binary images of bone
    [V1_bone,V2_bone] = BoneVolume(V1,V2,Bone_cutoff);
    
%     figure, volshow(V1_bone)
%     figure, volshow(V2_bone)

    % Creating combined image to view
    V_intersection = 1*single(V1_bone & V2_bone);
    V_pre_only = 50*single(xor(V1_bone, V2_bone) & V1_bone);
    V_post_only = 250*single(xor(V1_bone, V2_bone) & V2_bone);
    V_together = V_intersection + V_pre_only + V_post_only;
    
    diffdisplay(V_together);
end

function [V_jaw] = jaw_points(V_bone,front,back,left,right,top1,top2,bottom1,bottom2,dim)
% Volume using points (using lines) - sloped height - one jaw only

    % point 1,2,3,4 bottom; 5,6,7,8 top; clockwise from top
    
    %point = [x,y,z]; x-front to back,y-left to right(looking at face),z-bottom to top
    point1 = [front,left,bottom1];
    point2 = [back,left,bottom2];
    point3 = [back,right,point2(3)];
    point4 = [front,right,point1(3)];
    point5 = [point1(1:2),top1];
    point6 = [point2(1:2),top2];
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

    if dim == 0
        V_jaw = V_bone;
        for i = 1:557
            for j = 1:557
                isPart = ((j-point1(2))*m1d>m1n*(i-point1(1))) & ...
                         ((j-point2(2))*m2d>m2n*(i-point2(1))) & ...
                         ((j-point3(2))*m3d>m3n*(i-point3(1))) & ...
                         ((j-point4(2))*m4d>m4n*(i-point4(1)));
                V_jaw(i,j,:) = V_jaw(i,j,:)*isPart;
            end
        end

        for i = 1:557
            for k = 1:441
                isPart = ((k-point1(3))*m5d>m5n*(i-point1(1))) & ...
                         ((k-point5(3))*m6d<m6n*(i-point5(1))); 
                V_jaw(i,:,k) = V_jaw(i,:,k)*isPart;
            end
        end

        diffdisplay(V_jaw)
        
    else
        temp = size(V_bone);
        V_jaw_plane = zeros(6,temp(1),temp(2),temp(3));
        
        for i = 1:557
            k = uint16(((m5n*(i-point1(1)))/m5d)+point1(3));
            V_jaw_plane(1,i,:,k) = 99*ones(1,1,temp(2));
            k = uint16(((m6n*(i-point5(1)))/m6d)+point5(3));
            V_jaw_plane(2,i,:,k) = 99*ones(1,1,temp(2));
        end
        for i = 1:557
            for j = 1:557
                isPlane = (int16((j-point2(2))*m2d) == int16(m2n*(i-point2(1))));
                V_jaw_plane(3,i,j,:) = 99 * isPlane;
                isPlane = (int16((j-point4(2))*m4d) == int16(m4n*(i-point4(1))));
                V_jaw_plane(4,i,j,:) = 99 * isPlane;
                isPlane = (int16((j-point1(2))*m1d) == int16(m1n*(i-point1(1))));
                V_jaw_plane(5,i,j,:) = 99 * isPlane;
                isPlane = (int16((j-point3(2))*m3d) == int16(m3n*(i-point3(1))));
                V_jaw_plane(6,i,j,:) = 99 * isPlane;
            end
        end
        
        for i = 1:dim
            diffdisplay(squeeze(V_jaw_plane(i,:,:,:)) | V_bone)
        end
    end
end

function [] = diffdisplay(V)
    Transparency = reshape(linspace(1,1,256),256,1);
    Transparency(1) = 0;
    Colouring = winter(256);
    for i = 1:30
        Colouring(i,:) = [1,1,1]; %white
    end
    for i = 31:100
        Colouring(i,:) = [173/256,255/256,47/256]; %green
    end
    for i = 101:256
        Colouring(i,:) = [255/256,105/256,180/256]; %pink
    end
    figure; volshow(V,'Alphamap',Transparency,'Colormap',Colouring);
end

function [t1] = process_mask(V_mask)
    t1 = imclearborder(V_mask);
    figure,volshow(t1);
    t1 = imfill(t1, 'holes');
    figure,volshow(t1);
    se = strel('sphere',4);
    t1 = imerode(t1, se);
    figure,volshow(t1);
    t1 = imdilate(t1, se);
    t1 = imdilate(t1, se);
    figure,volshow(t1);
    se = strel('cuboid',[4 4 8]);
    t1 = imdilate(t1, se);
    t1 = imdilate(t1, se);
    t1 = imdilate(t1, se);
    t1 = imdilate(t1, se);
    t1 = imdilate(t1, se);
    figure,volshow(t1);
%     maskimg = V_pre_bottom;
%     maskimg(~t1) = 0;
%     figure,volshow(maskimg);
%     combine_bone(t2,t1,0.5);
% 
%     aaa = V_pre_bottom .* maskimg;
%     t1 = imfill(t1, 'holes');
%     bbb = V_post_bottom_alt_adj .* maskimg;
%     t1 = imfill(t1, 'holes');
end

function [geomtform,movingRegisteredVolume] = align_3d(fixedVolume,movingVolume,V_pre_info,V_post_info)

% DON'T USE, CANNOT DEBUG
% how to store header information? need struct with 
% SliceThickness: _  --- in DICOM header, 
% PixelSize: [_,_]   --- in DICOM header, PixelSpacings (0.3,0.3)

    [optimizer,metric] = imregconfig('monomodal');
    optimizer.MaximumIterations = 500;
    % optimizer.MaximumStepLength = optimizer.MaximumStepLength * 3;

    Rfixed   = imref3d(size( fixedVolume), V_pre_info.Scales(1), V_pre_info.Scales(2), V_pre_info.Scales(3));
    Rmoving  = imref3d(size(movingVolume),V_post_info.Scales(1),V_post_info.Scales(2),V_post_info.Scales(3));
    
    % movingRegisteredVolume = imregister(movingVolume,Rmoving,fixedVolume,Rfixed,'rigid',optimizer,metric);

    geomtform = imregtform(movingVolume,Rmoving,fixedVolume,Rfixed,'rigid',optimizer,metric)
    geomtform.T

    movingRegisteredVolume = imwarp(movingVolume,Rmoving,geomtform,'bicubic','OutputView',Rfixed);

    figure,imshowpair(movingRegisteredVolume(:,:,220), fixedVolume(:,:,220));
    title('Axial Slice of Registered Volume')

    helperVolumeRegistration(fixedVolume,movingRegisteredVolume);
end


function [volume] = Numerical_volume(V_bin)
% Finding volume of selected jaw (or any binary image). 
% Input is 3d binary image of jaw V_bin

volume_array = regionprops3(V_bin,'Volume');
volume = sum(volume_array.Volume);
end

