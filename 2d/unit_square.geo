// Gmsh project created on Sun Sep 29 20:48:15 2024
//+
Point(1) = {-0.5, 0.5, 0, 1.0};
//+
Point(2) = {0.5, -0.5, 0, 1.0};
//+
Point(3) = {0.5, 0.5, 0, 1.0};
//+
Point(4) = {-0.5, -0.5, 0, 1.0};
//+
Line(1) = {1, 4};
//+
Line(2) = {4, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
