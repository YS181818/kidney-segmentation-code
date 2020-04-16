function [imgW_1,imgW_2]=tpsWarpDemo_three2018(imgInFilename_1,imgInFilename_2,imgInFilename_3,mapFilename)
%% Get inputs
%imgInFilename_1:moving image
%imgInFilename_2:moving label
%imgInFilename_3:fixed label
interp.method = 'invdist'; %'nearest'; %'none' % interpolation method
interp.radius = 5; % radius or median filter dimension
interp.power = 2; %power for inverse wwighting interpolation method

imgIn_1 = imgInFilename_1;
imgIn_2 = imgInFilename_2;
[size_a,size_b,~]=size(imgIn_1);
load(mapFilename); % load map

%% Get the landmark points
if 0 % Get points interactively
    NPs = input('Enter number of landmark points : ');
    fprintf('Select %d correspondence / landmark points with mouse on Fig.2.\n',NPs);

    figure(2);
    Hp=subplot(1,2,1); % for landmark point selection
    image(imgIn);
    colormap(map);
    hold on;
    
    Hs=subplot(1,2,2); % for correspondence point selection
    imagesc(imgIn);
    colormap(map);
    hold on;
    
    Xp=[]; Yp=[]; Xs=[]; Ys=[];
    for ix = 1:NPs
        axis(Hp);
        [Yp(ix),Xp(ix)]=ginput(1); % get the landmark point
        scatter(Yp(ix),Xp(ix),32,'y','o','filled'); % display the point
        text(Yp(ix),Xp(ix),num2str(ix),'FontSize',6);
        
        axis(Hs);
        [Ys(ix),Xs(ix)]=ginput(1); % get the corresponding point
        scatter(Ys(ix),Xs(ix),32,'y','*'); % display the point
        text(Ys(ix),Xs(ix),num2str(ix),'FontSize',6);
    end
else%% get the ellipse vertexes of the kidney as 4 corresponding landmark points 
    [P1,P2,P3,P4]=computepixel(imgInFilename_2);
    Yp = [P1(1),P2(1),P3(1),P4(1)];
    Xp = [P1(2),P2(2),P3(2),P4(2)];  
    [P1,P2,P3,P4]=computepixel(imgInFilename_3);
    Ys = [P1(1),P2(1),P3(1),P4(1)];
    Xs = [P1(2),P2(2),P3(2),P4(2)]; 
end

%% Warping moving image
[imgW_1, imgWr_1]  = tpswarp(imgIn_1,[size(imgIn_1,2) size(imgIn_1,1)],[Xp' Yp'],[Xs' Ys'],interp); % thin plate spline warping
%% Warping moving label
[imgW_2, imgWr_2]  = tpswarp(imgIn_2,[size(imgIn_2,2) size(imgIn_2,1)],[Xp' Yp'],[Xs' Ys'],interp); % thin plate spline warping
imgW_2(find(imgW_2<0.5))=0;
imgW_2(find(imgW_2>0.5|imgW_2==0.5))=1;
imgW_2 = double(imgW_2);
imgWr_2 = double(imgWr_2);
%% Display
figure;
subplot(2,3,1); imshow(imgIn_1,[]);
for ix = 1 : length(Xp),
	impoint(gca,Yp(ix),Xp(ix));
end
% colormap(map);
subplot(2,3,2); imshow(imgWr_1,[]);
for ix = 1 : length(Xs),
	impoint(gca,Ys(ix),Xs(ix));
end
% colormap(map);
subplot(2,3,3); imshow(imgW_1,[]);
for ix = 1 : length(Xs),
	impoint(gca,Ys(ix),Xs(ix));
end
subplot(2,3,4); imshow(imgIn_2,[]);
for ix = 1 : length(Xp),
	impoint(gca,Yp(ix),Xp(ix));
end
% colormap(map);
subplot(2,3,5); imshow(imgWr_2,[]);
for ix = 1 : length(Xs),
	impoint(gca,Ys(ix),Xs(ix));
end
% colormap(map);
subplot(2,3,6); imshow(imgW_2,[]);
for ix = 1 : length(Xs),
	impoint(gca,Ys(ix),Xs(ix));
end
colormap(map);

