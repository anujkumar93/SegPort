function  score = reversescore(im,scores)

 
CROPPED_DIM = 480;

%store image score 
class_num =2;
score_index =zeros(size(im,1),size(im,2),class_num);
score =zeros(size(im,1),size(im,2),class_num);

indicesi = [0 size(im,1)-CROPPED_DIM] + 1;
indicesj = [0 size(im,2)-CROPPED_DIM] + 1;
curr = 1;
%four conner with mirror 8
for i = indicesi
  for j = indicesj
    score(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)= score(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)+scores(:,:,:,curr);
    score(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :) = score(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)+scores(:,end:-1:1,:,curr+5);
    score_index(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)= score_index(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)+2;
    curr = curr + 1;
  end
end
% middle with mirror 2
centeri = floor(indicesi(2) / 2)+1;
centerj = floor(indicesj(2) / 2)+1;
 
score(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)= ...
    score(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)+scores(:,:,:,5);
score(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)= ...
    score(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)+scores(:,end:-1:1,:,10);
score_index(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)= ...
    score_index(centeri:centeri+CROPPED_DIM-1,centerj:centerj+CROPPED_DIM-1,:)+2;
%tempscore=zeros(size(im,1),size(im,2),21,2); 
% for i=1:1:21
%     tempscore(:,:,i,1) = imresize(scores(:,:,i,11),[size(im,1),size(im,2)],'bilinear');
%     tempscore(:,:,i,2) = imresize(scores(:,:,i,12),[size(im,1),size(im,2)],'bilinear');
% end
% score =score +tempscore(:,:,:,1);
% score =score +tempscore(:,end:-1:1,:,2);
% score_index = score_index+2;
score =score./score_index;
end