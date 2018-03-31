function masknew = removesmall(mask)
 bw = im2bw(mask(:,:,1));
   cc = bwconncomp(bw);
   numpixels = cellfun(@numel,cc.PixelIdxList);
   maxpixels = max(numpixels(:));
   
   masknew = mask;
   for j=1:length(numpixels)
       if numpixels(j)<maxpixels
           masknew(cc.PixelIdxList{j}) = 0;
       end
   end
   masknew = double(masknew);
end