// Gmsh project created on Sun Sep 15 16:47:58 2024
//+
Point(1) = {-0, 0.5, -0, 1.0};
//+
Point(2) = {-0.5, 0.5, 0, 1.0};
//+
Recursive Delete {
  Point{2}; Point{1}; 
}
//+
Point(1) = {-0, 1, 0, 1.0};
//+
Point(2) = {-1, -0, 0, 1.0};
//+
Point(3) = {-0.5, -1, 0, 1.0};
//+
Point(4) = {0.5, -1, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};
//+
Physical Curve("TL", 6) = {1};
//+
Physical Curve("BL", 7) = {2};
//+
Physical Curve("B", 8) = {3};
//+
Physical Curve("BR", 9) = {4};
//+
Physical Curve("TR", 10) = {5};
//+
Curve Loop(1) = {1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Physical Surface("S", 11) = {1};
//+
Show "*";
