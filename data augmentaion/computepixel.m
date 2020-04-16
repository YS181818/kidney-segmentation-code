function [P1,P2,P3,P4] = computepixel(inputlabel)
%UNTITLED2 Summary of this function goes here
% modeling the kidney as ellipse
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

end

