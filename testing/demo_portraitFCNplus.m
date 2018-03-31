% demo portraitFCNplus model

clear
clc

addpath('../caffe-portraitseg/matlab/');

model_def_file  = './our_models/deploy_6channels.prototxt';
model_file      = './our_models/bgr_mmask_xy.caffemodel';

% caffe.set_mode_cpu();

caffe.set_mode_gpu();
caffe.set_device(3);

caffe.reset_all()
net = caffe.Net(model_def_file, model_file, 'test');

img = imread('00321.jpg');
load('../data/images_tracker/00321.mat');
[warpedxy warpedmask] = get_warped_xy_mmask(tracker);

img = single(img);
[h w chs] = size(img);

if chs==1
    img = repmat(img, [1 1 3]);
end

inputdata = zeros(h,w,6);
input_data(:,:,1) = (img(:,:,3)-104.008)/255;
input_data(:,:,2) = (img(:,:,2)-116.669)/255;
input_data(:,:,3) = (img(:,:,1)-122.675)/255;
input_data(:,:,4:5) = warpedxy;
input_data(:,:,6) = warpedmask;

input_data = permute(input_data,[2 1 3]);

[h1,w1,c1] = size(input_data);
net.blobs('data').reshape([h1,w1,c1,1]);
net.blobs('data').set_data(input_data);
net.forward_prefilled();
res = net.blobs('upscore').get_data();

diffs = exp(res(:,:,2)-res(:,:,1));
finalres = diffs./(1+diffs);
finalres = double(finalres'>0.5);
img = double(img/255);
figure,imshow([img img.*repmat(finalres, [1 1 3]) + 1- repmat(finalres, [1 1 3])])
