//+
Point(1) = {-0.7, 0.8, 0, 1.0};
//+
Point(2) = {-1.2, 0.5, 0, 1.0};
//+
Point(3) = {-1, 0.5, 0, 1.0};
//+
Point(4) = {-0.8, 0.6, 0, 1.0};
//+
Point(5) = {-0.7, 0.5, 0, 1.0};
//+
Point(6) = {-1.1, 0.7, 0, 1.0};
//+
Line(1) = {6, 3};
//+
Recursive Delete {
  Curve{1}; 
}
//+
Point(6) = {-0.9, 0.7, 0, 1.0};
//+
Line(1) = {2, 5};
//+
Line(2) = {5, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {1, 6};
//+
Line(5) = {6, 2};
//+
Curve Loop(1) = {4, 5, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("l1", 6) = {1};
//+
Physical Curve("l2", 7) = {2};
//+
Physical Curve("l3", 8) = {3};
//+
Physical Curve("l4", 9) = {4};
//+
Physical Curve("l5", 10) = {5};
//+
Physical Surface("S", 11) = {1};
