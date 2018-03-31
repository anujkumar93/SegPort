function [scores, maxlabel] = matcaffe_demo_ours(im, use_gpu)
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 


% init caffe network (spews logging info)

if exist('use_gpu', 'var')
  matcaffe_init_ours(use_gpu);
else
  matcaffe_init_ours();
end

%%input data directory
%dirs = dir('/media/hchen/Data/Medical/2015MICCAI_Gland_Seg/Validation_beign/*.bmp');
%dirs =struct2cell(dirs);
im_path = '/media/hchen/Data/Medical/2015MICCAI_Gland_Seg/Validation_beign/';
im_files = dir([im_path '*.bmp']);
%label_path = '/media/hchen/Data/Medical/2015MICCAI_Gland_Seg/Validation_benign_truth/';
%use VOC label map to store the image
%map = VOClabelcolormap;
%confusionmatrix = zeros(21,21);

for i=1:size(im_files,1)
  i
im = imread([im_path im_files(i).name]);
%%test data label image
%label = imread([label_path im_files(i).name]);
tic;
input_data = {prepare_image(im)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

scores = scores{1};
size(scores)
 
scores = permute(scores,[2 1 3 4]);
scores = squeeze(scores);
score =scores;
score = reversescore(im,scores);
% %********************************upsample scoremap*************************
% %tempscore = zeros(size(imlabel,1),size(imlabel,2),size(score,3));
% tempscore = zeros(size_im(1),size_im(2),size(score,3));
% for ll=1:1:size(score,3)
%    tempscore(:,:,ll) = imresize(score(:,:,ll),[size_im(1),size_im(2)],'bilinear');
% end
% score =tempscore;
% %save(sprintf('%s%s.mat',scorepath,dirs{1,i}),'score');
% %********************************upsample scoremap*************************


[~,maxlabel] = max(score,[],3);
%imwrite(maxlabel,map,sprintf('%s%s.png','./result/',dirs{1,i}(1:length(dirs{1,i})-4)));
imwrite(maxlabel,sprintf('%s%s.png','./result/',dirs{1,i}));
maxlabel = imread(sprintf('%s%s.png','./result/',dirs{1,i}));
maxlabel = maxlabel+1;
 end

%   for j =1:1:size(imlabel,1)
%     for l =1:1:size(imlabel,2)
%         if(imlabel(j,l)~=255)
%             confusionmatrix(imlabel(j,l)+1,maxlabel(j,l)) =  ...
%                confusionmatrix(imlabel(j,l)+1,maxlabel(j,l))+1;      
%         end
%     end
%  end
%   
%   result =zeros(1,21);
%   for i =1:1:21
%       tempa = sum(confusionmatrix(i,:));
%       tempb = sum(confusionmatrix(:,i));
%       result(i) = confusionmatrix(i,i)/(tempa+tempb-confusionmatrix(i,i));
%   end
%  save('confusion.mat','confusionmatrix');
%   save ('result.mat','result');
%  average  =sum(result(:))/21

% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
% five crop method 
CROPPED_DIM = 480;

im =single(im);
%im = im*0.0039;
mean_r = 204.5;
mean_g = 142;
mean_b = 203.9;
im(:,:,1) = im(:,:,1)-mean_r;
im(:,:,2) = im(:,:,2)-mean_g;
im(:,:,3) = im(:,:,3)-mean_b;

im = im(:,:,[3 2 1]);

images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
 
indicesi = [0 size(im,1)-CROPPED_DIM] + 1;
indicesj = [0 size(im,2)-CROPPED_DIM] + 1;
curr = 1;
%map =VOClabelcolormap;
for i = indicesi
  for j = indicesj
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
centeri = floor(indicesi(2) / 2)+1;
centerj = floor(indicesj(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
