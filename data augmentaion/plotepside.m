function [ ] = plotepside(inputimage,inputlabel)
%UNTITLED Summary of this function goes here
% modeling the kidney as ellipse and plot the ellipse
s=regionprops(inputlabel,'BoundingBox','Centroid','MajorAxisLength'...
                               ,'MinorAxisLength','Eccentricity','Orientation','ConvexHull');
theta=(s.Orientation/180)*pi;
cx_pixel = s.Centroid(1);
cy_pixel = s.Centroid(2);
%%% left
P1_x=cx_pixel-0.5*s.MajorAxisLength*cos(theta);
P1_y=cy_pixel+0.5*s.MajorAxisLength*sin(theta);
%%% up
P2_x=cx_pixel-0.5*s.MinorAxisLength*cos(pi/2-theta);
P2_y=cy_pixel-0.5*s.MinorAxisLength*sin(pi/2-theta);
%%% right
P3_x=cx_pixel+0.5*s.MajorAxisLength*cos(theta);
P3_y=cy_pixel-0.5*s.MajorAxisLength*sin(theta);
%%% down
P4_x=cx_pixel+0.5*s.MinorAxisLength*cos(pi/2-theta);
P4_y=cy_pixel+0.5*s.MinorAxisLength*sin(pi/2-theta);
P1=[P1_x,P1_y];
P2=[P2_x,P2_y];
P3=[P3_x,P3_y];
P4=[P4_x,P4_y];
a=s.MajorAxisLength/2; b=s.MinorAxisLength/2;
S=zeros(321,321);
for x=1:321
    for y=1:321
       if ((x-cx_pixel)*cos(-theta)+(y-cy_pixel)*sin(-theta))^2/(a^2)+((x-cx_pixel)*sin(-theta)-(y-cy_pixel)*cos(-theta))^2/b^2<=1;
           S(y,x)=1;
       end 
    end
end
imshow(inputimage,[]);
hold on;
contour(S, [0.5,0.5], 'color', 'y', 'LineWidth', 2);
hold on;
contour(inputlabel, [0.5,0.5], 'color', 'r', 'LineWidth', 2);
hold on;
plot(P1(1),P1(2),'y*');
hold on;
plot(P2(1),P2(2),'y*');
hold on;
plot(P3(1),P3(2),'y*');
hold on;
plot(P4(1),P4(2),'y*');
end

