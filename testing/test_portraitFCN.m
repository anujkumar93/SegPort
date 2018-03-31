% test portraitFCN model

% test portraitFCN+ model
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

% load the test image list
load('../data/testlist.mat')
allIoU = 0;
numimg = 0;
for i=1:length(testlist)
    disp(['To test image ' sprintf('%05d',testlist(i))]);
    
    if exist(['../data/portraitFCN_data/' sprintf('%05d',testlist(i)) '.mat'],'file')
        load(['../data/portraitFCN_data/' sprintf('%05d',testlist(i)) '.mat']);
        load(['../data/images_mask/' sprintf('%05d',testlist(i)) '_mask.mat']);
        numimg = numimg + 1;
        input_data = single(img);
        input_data = permute(input_data,[2 1 3]);
        
        [h1,w1,c1] = size(input_data);
        net.blobs('data').reshape([h1,w1,c1,1]);
        net.blobs('data').set_data(input_data);
        net.forward_prefilled();
        res = net.blobs('upscore').get_data();
        
        diffs = exp(res(:,:,2)-res(:,:,1));
        finalres = diffs./(1+diffs);
        finalres = double(finalres'>0.5);
        allIoU = allIoU + sum(finalres(:).*mask(:))/sum(double((finalres(:)+mask(:))>0));
    end
end
meanIoU = allIoU/numimg;

disp(['mean IoU is ' num2str(meanIoU*100) '% in ' num2str(numimg) ' test images']);
