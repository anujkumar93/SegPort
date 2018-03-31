function [warped_xy warped_mmask] = get_warped_xy_mmask(destracker)

load('../data/images_tracker/00047.mat')
reftracker = tracker;
refpos = floor(mean(reftracker));
[xxc yyc] = meshgrid(1:1800,1:2000);
% normalize the x- and y-channels
xxc = (xxc-600-refpos(1))/600;
yyc = (yyc-600-refpos(2))/800;

maskc = im2double(imread('../data/meanmask.png'));
maskc = padarray(maskc,[600 600]);

if size(destracker,1)==49
    
    [tform,~,~] = estimateGeometricTransform(double(reftracker)+repmat([600 600],[49 1]),...
        double(destracker)+repmat([600 600],[49 1]),'affine');
    outputView = imref2d(size(xxc));
    warpedxx = imwarp(xxc,tform,'OutputView',outputView);
    warpedyy = imwarp(yyc,tform,'OutputView',outputView);
    warpedmask = imwarp(maskc,tform,'OutputView',outputView);
    
    warpedxx = warpedxx(601:1400,601:1200,:);
    warpedyy = warpedyy(601:1400,601:1200,:);
    warped_xy = cat(3,warpedxx,warpedyy);
    warped_mmask = warpedmask(601:1400,601:1200,:);
else
    warped_xy = zeros(800,600,2);
    warped_mmask = zeros(800,600);
end
