video = VideoWriter('yourvideo.avi'); %create the blank video object
open(video); %open the empty video
for ii=1:1:500 %where 3 is the number of images; iterative placing of image based on file name
  N=ii; %i happened to name my tif's in intervals of 100 so i added this
  I = ['Ph_num=10_iter=',int2str(N),'.bmp']; %read the next image
  A = imread(I);
  text = "Iteration: "+num2str(ii);
  A = insertText(A, [0 0],text,FontSize=36);
  
  writeVideo(video,A); %write the image to video
end
%% 

close(video); %close the file
%% 


% loads the video. insert the location of your video from above
obj = VideoReader('D:\Darkroom\Joshua\157 final project\sbmirresults\yourvideo.avi');
  
% Write in new variable
obj2= VideoWriter('video ampli.avi');    
  
% decrease framerate 
obj2.FrameRate = 10;              
open(obj2);
  
% for reading frames one by one
while hasFrame(obj)              
    k = readFrame(obj); 
  
    % write the frames in obj2.         
    obj2.writeVideo(k);          
end
  
close(obj2);