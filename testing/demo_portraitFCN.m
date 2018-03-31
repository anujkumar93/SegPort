% demo portraitFCN model

clear
clc

addpath('../caffe-portraitseg/matlab/');

model_def_file  = './our_models/deploy_3channels.prototxt';
model_file      = './our_models/bgr.caffemodel';

% caffe.set_mode_cpu();

caffe.set_mode_gpu();
caffe.set_device(3);

caffe.reset_all()
net = caffe.Net(model_def_file, model_file, 'test');

img = imread('00321.jpg');

img = single(img);
input_data = img;
input_data(:,:,1) = img(:,:,3)-104.008;
input_data(:,:,2) = img(:,:,2)-116.669;
input_data(:,:,3) = img(:,:,1)-122.675;
input_data = input_data/255;

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
